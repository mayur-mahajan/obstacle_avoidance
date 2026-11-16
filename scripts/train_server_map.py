"""Train the server's obstacle map (carsim/server_map.py) on the cached overhead-picture features.

Data: runs/overhead_native2 train + train_heading (48k pictures of 9000 random rooms) -> val_rooms (1000 pictures of
500 other rooms) and val_drive (600 pictures of the 27 validation driving rooms). Labels come from each room's seed.
usage: python scripts/train_server_map.py [--steps 6000] [--out runs/checkpoints/server_map_s0.pt]"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

DATA = ROOT / "runs/overhead_native2"


def room_labels(seeds, cache):
    """seed -> (80, 80) blocked map, cached in one .npz (sampling a room takes ~20 ms)."""
    from carsim.server_map import label_map
    from carsim.world import EnvConfig, Grid, sample_episode
    have = dict(np.load(cache)) if cache.exists() else {}
    todo = [s for s in sorted(set(int(s) for s in seeds)) if str(s) not in have]
    if todo:
        grid, t0 = Grid(EnvConfig()), time.time()
        for k, s in enumerate(todo):
            have[str(s)] = label_map(sample_episode(s), grid)
            if k % 1000 == 0:
                print(f"labels {k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(cache, **have)
    return have


class Pics(torch.utils.data.Dataset):
    def __init__(self, splits, labels):
        self.parts = []
        for s in splits:
            f = np.load(DATA / s / "feats.npy", mmap_mode="r")
            seeds = np.load(DATA / s / "meta.npz")["seed"]
            self.parts.append((f, seeds))
        self.index = [(p, i) for p, (f, _s) in enumerate(self.parts) for i in range(len(f))]
        self.labels = labels

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        p, i = self.index[k]
        f, seeds = self.parts[p]
        return torch.from_numpy(np.array(f[i])), torch.from_numpy(self.labels[str(int(seeds[i]))])


@torch.no_grad()
def evaluate(model, loader, dev):
    from carsim.server_map import map_loss, map_scores
    model.eval(); L, S = [], []
    for x, y in loader:
        with torch.autocast("cuda", torch.bfloat16):
            lg = model(x.to(dev))
        lg = lg.float(); y = y.to(dev)
        L.append(map_loss(lg, y).item()); S.append(map_scores(lg, y))
    model.train()
    return {"loss": float(np.mean(L)), **{k: float(np.mean([s[k] for s in S])) for k in S[0]}}


if __name__ == "__main__":
    from carsim.server_map import MapHead, map_loss, map_scores
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/checkpoints/server_map_s0.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed); dev = "cuda"
    splits = {"train": ["train", "train_heading"], "val_rooms": ["val_rooms"], "val_drive": ["val_drive"]}
    seeds = np.concatenate([np.load(DATA / s / "meta.npz")["seed"] for v in splits.values() for s in v])
    labels = room_labels(seeds, DATA / "map_labels.npz")
    print("blocked share of cells: %.3f" % np.mean([m.mean() for m in labels.values()]), flush=True)
    mk = lambda names, shuffle: torch.utils.data.DataLoader(Pics(names, labels), batch_size=a.batch, shuffle=shuffle,
                                                          num_workers=6, drop_last=shuffle, persistent_workers=shuffle)
    train, vr, vd = mk(splits["train"], True), mk(splits["val_rooms"], False), mk(splits["val_drive"], False)
    model = MapHead().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = lambda s: min(1.0, s / 300) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / a.steps)))
    print("parameters %.1fM, train pictures %d" % (sum(p.numel() for p in model.parameters()) / 1e6, len(train.dataset)), flush=True)
    step, t0, acc, log = 0, time.time(), [], []
    while step < a.steps:
        for x, y in train:
            for g in opt.param_groups:
                g["lr"] = a.lr * sched(step)
            with torch.autocast("cuda", torch.bfloat16):
                lg = model(x.to(dev, non_blocking=True))
            y = y.to(dev)
            loss = map_loss(lg.float(), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            acc.append((loss.item(), map_scores(lg.float(), y)))
            if step % 250 == 0 or step == a.steps:
                row = {"step": step, "train_loss": float(np.mean([l for l, _ in acc])),
                       **{"train_" + k: float(np.mean([s[k] for _, s in acc])) for k in acc[0][1]}, "minutes": (time.time() - t0) / 60}
                acc = []
                if step % 1000 == 0 or step == a.steps:
                    row.update({"val_rooms_" + k: v for k, v in evaluate(model, vr, dev).items()})
                    row.update({"val_drive_" + k: v for k, v in evaluate(model, vd, dev).items()})
                log.append(row)
                print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
            if step >= a.steps:
                break
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "log": log, "args": vars(a) | {"out": str(a.out)}}, a.out)
    print("saved", a.out, flush=True)
