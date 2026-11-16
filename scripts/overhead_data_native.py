"""Practice pictures for the NATIVE SmolVLM2 server (splits train / val_rooms / val_drive copy the scenes of
scripts/overhead_data.py; train_heading / val_heading are new "arrow practice" scenes: each spot at 4 headings).

Original description: practice pictures for the NATIVE SmolVLM2 server (notebook 06): the SAME scenes as scripts/overhead_data.py (same rooms,
car poses and camera jitter, read from its meta.npz) re-rendered at 1024 px and encoded the way SmolVLM2 itself does it:
4 tiles of 512 px + the whole picture shrunk to 512 ("image splitting"), each -> 64 tokens = (5, 64, 960) per picture.

usage: python scripts/overhead_data_native.py [--src runs/overhead_pretrain] [--out runs/overhead_native] [--splits ...]"""
import argparse, os, sys, time
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

RES = 1024
TILES = 5          # 2 x 2 tiles + the global view


def main():
    from carsim import overhead
    from carsim.config import CarConfig
    from carsim.env import NavEnv
    from carsim.server_native import load_smolvlm2
    from carsim.world import sample_episode

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "runs/overhead_pretrain")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/overhead_native")
    ap.add_argument("--splits", nargs="+", default=["val_drive", "val_rooms", "train"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N pictures of each split (smoke test)")
    ap.add_argument("--marker", choices=["cyan", "two"], default="two", help="roof marker: all cyan (first design) or pink front + cyan back")
    ap.add_argument("--heading_rooms", type=int, default=3000, help="rooms for the train_heading split (x 2 spots x 4 headings)")
    a = ap.parse_args()

    env = NavEnv(car_cfg=overhead.car_config(two_colour=a.marker == "two"), overview_camera_res=(RES, RES), backend="cpu")
    model, proc = load_smolvlm2("cuda")

    @torch.no_grad()
    def encode(pics):
        from PIL import Image
        pv = proc.image_processor([[Image.fromarray(p)] for p in pics], return_tensors="pt")
        x = pv["pixel_values"].to("cuda", torch.bfloat16)                      # (B, 5, 3, 512, 512)
        assert x.shape[1] == TILES, x.shape
        m = pv["pixel_attention_mask"].to("cuda")
        f = model.model.get_image_features(x, m, return_dict=True).pooler_output   # (B*5, 64, 960), or a list of them
        if isinstance(f, (list, tuple)):
            f = torch.cat(list(f), 0)
        return f.reshape(len(pics), TILES, 64, -1).half().cpu().numpy()

    def heading_meta(n_rooms, positions, headings, lo, seed):
        """Arrow practice: each random spot is shown with the car at `headings` evenly spaced headings (random start),
        so where the car stands never gives away the answer - only the roof arrow does. New rooms (seeds lo..lo+1e9)."""
        rng = np.random.default_rng(seed)
        meta = {k: [] for k in ["seed", "pose", "jitter", "objects"]}
        for s in rng.choice(np.arange(lo, lo + 1_000_000_000), n_rooms, replace=False):
            spec = sample_episode(int(s)); env.reset(spec=spec)
            for _ in range(positions):
                x, y, yaw0 = overhead.random_free_pose(env, spec, rng)
                for h in range(headings):
                    yaw = (yaw0 + 2 * np.pi * h / headings + np.pi) % (2 * np.pi) - np.pi
                    meta["seed"].append(int(s)); meta["pose"].append((x, y, yaw))
                    meta["jitter"].append(overhead.sample_jitter(rng)); meta["objects"].append(overhead.object_table(spec))
        return {k: np.array(v) for k, v in meta.items()}

    for split in a.splits:
        if split == "train_heading":
            meta = heading_meta(a.heading_rooms, 2, 4, 5_000_000_000, 11)
        elif split == "val_heading":
            meta = heading_meta(250, 1, 4, 6_000_000_000, 12)
        else:
            meta = np.load(a.src / split / "meta.npz")
        meta = {k: meta[k] for k in (meta.files if hasattr(meta, "files") else meta)}
        N = len(meta["seed"]) if a.limit is None else min(a.limit, len(meta["seed"]))
        d = a.out / split; d.mkdir(parents=True, exist_ok=True)
        feats = np.lib.format.open_memmap(d / "feats.npy", mode="w+", dtype=np.float16, shape=(N, TILES, 64, 960))
        buf, preview, cur, t0 = [], [], None, time.time()
        for n in range(N):
            seed = int(meta["seed"][n])
            if seed != cur:
                env.reset(spec=sample_episode(seed)); cur = seed
            env.car.reset(*meta["pose"][n])
            pic = overhead.render(env, meta["jitter"][n])
            buf.append(pic)
            if len(preview) < 8:
                preview.append(pic)
            if len(buf) == 16 or n == N - 1:
                feats[n + 1 - len(buf): n + 1] = encode(buf); buf = []
            if n % 2000 == 0:
                print(f"{split} {n}/{N} {time.time() - t0:.0f}s", flush=True)
        feats.flush()
        np.savez(d / "meta.npz", **{k: meta[k][:N] for k in meta})             # same scenes, same labels
        np.save(d / "preview.npy", np.stack(preview))
        print(f"{split} done: {N} pictures, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
