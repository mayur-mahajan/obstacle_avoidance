"""Overhead view + server outputs for EVERY frame of a driving dataset (notebook 06).

For each frame of the training file (e.g. runs/datasets/route_follow_prep.json), using the car pose and room seed the
recorder stored: render the overhead camera (two-colour roof marker, random camera jitter), encode it the SmolVLM2 way
(4 tiles + whole view = 5 x 64 tokens), and run the trained, FROZEN server on it with the frame's instruction.
Saved to OUT/ (one row per frame, in the training table's order):
  feats.npy      (N, 5, 64, 960) fp16   overhead picture tokens       -> the "full overhead picture" comparison model
  message.npy    (N, 8, 128)     fp16   the server's message          -> the main design (2 KB per frame)
  facts_pred.npy (N, 13)         fp32   the server's decoded facts    -> option C, realistic (a few numbers)
  facts_true.npy (N, 9)          fp32   the simulator's true facts    -> perfect-message warm-up / option C upper bound
  meta.npz       pose, jitter, seed, row index
Facts layout (9): car x, y (/4 m), heading cos, sin, goal x, y (/4 m), goal direction from the car cos, sin, distance (/4 m);
facts_pred adds the 4 ahead/left/behind/right scores.

usage: python scripts/overhead_frames.py --prep runs/datasets/route_follow_prep.json --server runs/checkpoints/server_native2_u4_heading_s0.pt
                                         --out runs/overhead_frames/route_follow"""
import argparse, os, sys, time
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch


def main():
    from carsim import overhead
    from carsim.data import load_training_data
    from carsim.env import NavEnv
    from carsim.server import FACTS, XY_SCALE
    from carsim.server_native import NativeServerConfig, NativeServerTrainer
    from carsim.world import sample_episode

    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", type=Path, default=ROOT / "runs/datasets/route_follow_prep.json")
    ap.add_argument("--server", type=Path, default=ROOT / "runs/checkpoints/server_native2_u4_heading_s0.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/overhead_frames/route_follow")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="camera jitter seed")
    a = ap.parse_args()

    _prep, table, _frames, _eps = load_training_data(a.prep)
    N = len(table) if a.limit is None else a.limit
    env = NavEnv(car_cfg=overhead.car_config(two_colour=True), overview_camera_res=(1024, 1024), backend="cpu")
    # the frozen server (reader + compressor + decoder) exactly as trained
    ck = torch.load(a.server, map_location="cpu", weights_only=False)
    cfg = NativeServerConfig(**{k: v for k, v in ck["cfg"].items() if k in NativeServerConfig.__dataclass_fields__})
    cfg.init, cfg.train_splits = str(a.server), ("val_drive",)
    srv = NativeServerTrainer(cfg, ROOT / "runs/overhead_native2")
    srv.reader.eval(); srv.comp.eval(); srv.dec.eval()
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    processor.image_processor.size = {"longest_edge": 1024}
    model = srv.reader.model

    a.out.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(a.out / "feats.npy", mode="w+", dtype=np.float16, shape=(N, 5, 64, 960))
    message = np.zeros((N, 8, 128), np.float16)
    facts_pred = np.zeros((N, sum(FACTS.values())), np.float32)
    facts_true = np.zeros((N, 9), np.float32)
    poses, jitters, seeds = np.zeros((N, 3), np.float32), np.zeros((N, 6), np.float32), np.zeros(N, np.int64)
    rng = np.random.default_rng(a.seed)

    @torch.no_grad()
    def flush(idx, pics, prompts):
        from PIL import Image
        pv = processor.image_processor([[Image.fromarray(p)] for p in pics], return_tensors="pt")
        f = model.model.get_image_features(pv["pixel_values"].to("cuda", torch.bfloat16), pv["pixel_attention_mask"].to("cuda"),
                                           return_dict=True).pooler_output
        f = (torch.cat(list(f), 0) if isinstance(f, (list, tuple)) else f).reshape(len(pics), 5, 64, -1)
        feats[idx] = f.half().cpu().numpy()
        msg, out = srv.forward(f.float(), prompts, torch.zeros(len(pics), dtype=torch.long))
        message[idx] = msg.half().cpu().numpy()
        facts_pred[idx] = torch.cat([out[k] for k in FACTS], 1).float().cpu().numpy()

    cur, buf, t0 = None, [], time.time()
    rows = table.iloc[:N]
    for n, (_, r) in enumerate(rows.iterrows()):
        seed = int(r["sim.episode_seed"])
        if seed != cur:
            spec = sample_episode(seed); env.reset(spec=spec); cur = seed
        pose = (float(r["sim.pose.x"]), float(r["sim.pose.y"]), float(r["sim.pose.yaw"]))
        env.car.reset(*pose)
        j = overhead.sample_jitter(rng)
        pic = overhead.render(env, j)
        _s, _c, gx, gy = spec.goal
        f = overhead.facts(np.array(pose), np.array([gx, gy]))
        facts_true[n] = np.concatenate([f["car_xy"] / XY_SCALE, f["heading"], f["obj_xy"] / XY_SCALE, f["rel_dir"], f["rel_dist"] / XY_SCALE])
        poses[n], jitters[n], seeds[n] = pose, j, seed
        buf.append((n, pic, r["task"]))
        if len(buf) == 16 or n == N - 1:
            flush([b[0] for b in buf], [b[1] for b in buf], [b[2] for b in buf]); buf = []
        if n % 2000 == 0:
            print(f"{n}/{N} {time.time() - t0:.0f}s", flush=True)
    feats.flush()
    np.save(a.out / "message.npy", message); np.save(a.out / "facts_pred.npy", facts_pred); np.save(a.out / "facts_true.npy", facts_true)
    np.savez(a.out / "meta.npz", pose=poses, jitter=jitters, seed=seeds, row=rows.index.to_numpy(), server=str(a.server), prep=str(a.prep))
    print(f"done {N} frames {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
