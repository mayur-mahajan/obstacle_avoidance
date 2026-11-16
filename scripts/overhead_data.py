"""Practice pictures for the SERVER's reader (notebook 06): overhead views of random rooms, labelled by the simulator.

No driving is needed: each room gets a few random car poses. Every picture is passed once through smolVLA's FROZEN
picture encoder (SigLIP + connector) and only its 64 tokens are saved (fp16, ~120 KB each instead of a 770 KB picture),
together with the car pose, camera jitter and the room's object table (carsim.overhead.object_table), so the training
question can be chosen fresh every time.

Splits written to OUT/<split>/{feats.npy, meta.npz, preview.npy}:
  train       random rooms (seeds 3e9..4e9)                   N_TRAIN_ROOMS x POSES pictures
  val_rooms   other random rooms (seeds 4e9..5e9)              500 x 2
  val_drive   frames of the 27 validation DRIVES (notebook 03 split): real driving poses, the rooms the car is tested on

usage: python scripts/overhead_data.py [--out runs/overhead_pretrain] [--train_rooms 6000] [--poses 4]"""
import argparse, os, sys, time, zlib
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def main():
    from carsim import overhead
    from carsim.config import CarConfig
    from carsim.data import load_training_data
    from carsim.env import NavEnv
    from carsim.vla import apply_reader_fix, load_smolvla
    from carsim.world import sample_episode

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs/overhead_pretrain")
    ap.add_argument("--train_rooms", type=int, default=6000)
    ap.add_argument("--poses", type=int, default=4)
    ap.add_argument("--val_rooms", type=int, default=500)
    ap.add_argument("--val_drive", type=int, default=600)
    a = ap.parse_args()

    env = NavEnv(car_cfg=CarConfig(roof_marker=True), overview_camera_res=(overhead.RES, overhead.RES), backend="cpu")
    policy = load_smolvla("cuda")
    apply_reader_fix(policy)
    vw = policy.model.vlm_with_expert

    @torch.no_grad()
    def encode(pics):   # (B, H, W, 3) uint8 -> (B, 64, 960) fp16: what embed_image returns (after the reader fix)
        from lerobot.policies.common.vla_utils import resize_with_pad
        x = torch.from_numpy(np.stack(pics)).cuda().permute(0, 3, 1, 2).float().div(255)
        x = resize_with_pad(x, 512, 512, pad_value=0) * 2 - 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return vw.embed_image(x).half().cpu().numpy()

    def write(split, jobs):
        """jobs: list of (seed, pose or None). None = random free pose."""
        d = a.out / split
        d.mkdir(parents=True, exist_ok=True)
        N = len(jobs)
        feats = np.lib.format.open_memmap(d / "feats.npy", mode="w+", dtype=np.float16, shape=(N, 64, 960))
        meta = {k: [] for k in ["seed", "pose", "jitter", "objects"]}
        rng = np.random.default_rng(zlib.crc32(split.encode()))   # fixed per split (hash() changes every run)
        buf, preview, cur, t0 = [], [], None, time.time()
        for n, (seed, pose) in enumerate(jobs):
            if seed != cur:
                spec = sample_episode(int(seed)); env.reset(spec=spec); cur = seed
            pose = pose if pose is not None else overhead.random_free_pose(env, spec, rng)
            env.car.reset(*pose)
            j = overhead.sample_jitter(rng)
            pic = overhead.render(env, j)
            buf.append(pic)
            if len(preview) < 32:
                preview.append(pic)
            meta["seed"].append(seed); meta["pose"].append(pose); meta["jitter"].append(j)
            meta["objects"].append(overhead.object_table(spec))
            if len(buf) == 64 or n == N - 1:
                feats[n + 1 - len(buf): n + 1] = encode(buf); buf = []
            if n % 2000 == 0:
                print(f"{split} {n}/{N} {time.time() - t0:.0f}s", flush=True)
        feats.flush()
        np.savez(d / "meta.npz", **{k: np.array(v) for k, v in meta.items()})
        np.save(d / "preview.npy", np.stack(preview))
        print(f"{split} done: {N} pictures, {time.time() - t0:.0f}s", flush=True)

    rng = np.random.default_rng(0)
    tr = rng.choice(np.arange(3_000_000_000, 4_000_000_000), a.train_rooms, replace=False)
    va = rng.choice(np.arange(4_000_000_000, 5_000_000_000), a.val_rooms, replace=False)
    prep, table, _f, _e = load_training_data(ROOT / "runs/datasets/scripted_close_prep.json")
    vrows = table[table.episode_index.isin(set(prep["split"]["val"]))].sample(a.val_drive, random_state=0).sort_values("sim.episode_seed")
    write("val_drive", [(int(r["sim.episode_seed"]), (float(r["sim.pose.x"]), float(r["sim.pose.y"]), float(r["sim.pose.yaw"])))
                        for _, r in vrows.iterrows()])
    write("val_rooms", [(int(s), None) for s in va for _ in range(2)])
    write("train", [(int(s), None) for s in tr for _ in range(a.poses)])


if __name__ == "__main__":
    main()
