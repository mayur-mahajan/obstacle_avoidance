"""The SERVER side of notebook 06: a reader for the overhead picture + a compressor that squeezes what it read into a
small message for the car.

    overhead picture (64 tokens, pre-computed by scripts/overhead_data.py) + instruction
        -> server reader: smolVLA's SmolVLM2 (16 layers, our reader input format), last N language layers trainable
        -> compressor: 8 learned queries attend over the reader's outputs -> MESSAGE (8 x 128 numbers, ~2 KB in fp16)
        -> facts decoder (training only, and option C): car x/y + heading, named object x/y, object direction + distance
           as seen from the car

Why train it at all: the frozen reader could NOT find the named object on the overhead picture (visibility check,
2026-11-14: object position barely better than guessing; ahead/left/right/behind at chance). So the server first
practises on thousands of random rooms that the simulator labels for free (pre-training), before any driving.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import overhead

FACTS = {"car_xy": 2, "heading": 2, "obj_xy": 2, "rel_dir": 2, "rel_dist": 1, "dir4": 4}   # outputs of the decoder
XY_SCALE = 4.0                                                                           # metres -> about +-1


class Compressor(nn.Module):
    """K learned queries read the reader's output tokens (one cross-attention block + MLP, Perceiver-style), then each
    is shrunk to ``msg_dim`` numbers. The K x msg_dim result is the message sent to the car."""

    def __init__(self, dim: int = 960, k: int = 8, d: int = 256, msg_dim: int = 128, heads: int = 4):
        super().__init__()
        self.kv = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d))
        self.query = nn.Parameter(torch.randn(1, k, d) / math.sqrt(d))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, msg_dim))

    def forward(self, tokens, mask):
        kv = self.kv(tokens)
        q, w = self.attn(self.query.expand(len(kv), -1, -1), kv, kv, key_padding_mask=~mask)
        q = q + self.mlp(q)
        return self.out(q), w


class FactsDecoder(nn.Module):
    """Message -> the simple facts. During pre-training it forces the message to carry them; in option C its output
    IS what the server sends."""

    def __init__(self, k: int = 8, msg_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(1), nn.Linear(k * msg_dim, 512), nn.GELU(), nn.Linear(512, sum(FACTS.values())))

    def forward(self, msg):
        out, i = {}, 0
        y = self.net(msg)
        for name, n in FACTS.items():
            out[name] = y[:, i:i + n]; i += n
        return out


class ServerReader(nn.Module):
    """smolVLA's reader run on (overhead tokens, instruction) -> output tokens. Uses smolVLA's own prefix code with the
    picture encoder skipped (its output is pre-computed), so the overhead picture is presented exactly like a camera."""

    def __init__(self, policy, unlock_layers: int = 4):
        super().__init__()
        self.policy = policy
        self.unlock_layers = unlock_layers
        for p in policy.parameters():
            p.requires_grad = False
        self.layers = list(policy.model.vlm_with_expert.get_vlm_model().text_model.layers[-unlock_layers:]) if unlock_layers else []
        for L in self.layers:
            for p in L.parameters():
                p.data = p.data.float(); p.requires_grad = True   # fp32 master copies, bf16 compute

    def trainable(self):
        return [p for L in self.layers for p in L.parameters()]

    def forward(self, feats, prompts):
        from .vla import _prefix_forward, prepare_inputs
        m, vw = self.policy.model, self.policy.model.vlm_with_expert
        B, dev = len(feats), feats.device
        dummy = torch.zeros(B, 3, 8, 8, device=dev)                       # placeholder; the encoder is skipped below
        _imgs, img_masks, tok, tok_mask, state = prepare_inputs(self.policy, dummy, prompts, torch.zeros(B, 3), device=dev,
                                                                 fixed_length=True)
        keep = vw.embed_image
        vw.embed_image = lambda _x: feats.to(torch.bfloat16 if dev.type == "cuda" else feats.dtype)   # pre-computed tokens
        try:
            out, pad = _prefix_forward(m, [dummy], img_masks, tok, tok_mask, state)
        finally:
            vw.embed_image = keep
        return out[:, :-1].float(), pad[:, :-1]                           # drop the (unused) state token


# -------------------------------------------------------------------------------------------------
# Practice data: pre-computed overhead tokens + the room's objects; the question is picked fresh each time
# -------------------------------------------------------------------------------------------------
class OverheadQuestions(torch.utils.data.Dataset):
    """One item = (overhead tokens, instruction naming one object of the room, the facts for that object).
    train: a random object and wording every time; val: fixed (seeded) so scores are comparable between runs.
    The held-out goal (orange sphere, notebook 03) is never asked in training."""

    def __init__(self, root: Path, train: bool, seed: int = 0):
        from .data import Paraphraser
        self.feats = np.load(Path(root) / "feats.npy", mmap_mode="r")
        m = np.load(Path(root) / "meta.npz")
        self.pose, self.objects = m["pose"].astype(np.float32), m["objects"]
        self.train, self.seed = train, seed
        self.para = Paraphraser("train", seed)
        if not train:
            rng = np.random.default_rng(seed)
            self.fixed = [self._pick(i, rng) for i in range(len(self.pose))]

    def _pick(self, i, rng):
        from .world import COLORS, SHAPES
        objs = self.objects[i]
        ok = [k for k in range(len(objs)) if objs[k, 4] > 0
              and not (self.train and SHAPES[int(objs[k, 0])] == "sphere" and list(COLORS)[int(objs[k, 1])] == "orange")]
        k = int(rng.choice(ok))
        self.para.rng = rng
        return k, self.para(list(COLORS)[int(objs[k, 1])], SHAPES[int(objs[k, 0])])

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, i):
        k, prompt = self._pick(i, np.random.default_rng()) if self.train else self.fixed[i]
        f = overhead.facts(self.pose[i], self.objects[i, k, 2:4])
        target = np.concatenate([f["car_xy"] / XY_SCALE, f["heading"], f["obj_xy"] / XY_SCALE, f["rel_dir"],
                                 f["rel_dist"] / XY_SCALE]).astype(np.float32)
        return torch.from_numpy(np.array(self.feats[i])), prompt, torch.from_numpy(target), int(overhead.direction_class(f["bearing"]))


def _collate(items):
    f, p, t, c = zip(*items)
    return torch.stack(f), list(p), torch.stack(t), torch.tensor(c)


# -------------------------------------------------------------------------------------------------
# Pre-training
# -------------------------------------------------------------------------------------------------
@dataclass
class ServerConfig:
    steps: int = 6000
    batch: int = 64
    seed: int = 0
    unlock_layers: int = 4        # last N language layers of the server's reader that may change (0 = reader frozen)
    reader_lr: float = 3e-5       # smaller than the new parts: adapt the reader, don't overwrite it
    lr: float = 3e-4              # compressor + facts decoder (new, small)
    warmup: int = 300
    k: int = 8                    # message = k x msg_dim numbers
    msg_dim: int = 128
    log_every: int = 50
    val_every: int = 500


class ServerTrainer:
    def __init__(self, cfg: ServerConfig, data_root, device: str = "cuda"):
        from .vla import apply_reader_fix, load_smolvla
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.cfg, self.dev, root = cfg, torch.device(device), Path(data_root)
        policy = load_smolvla(device)
        apply_reader_fix(policy)
        self.reader = ServerReader(policy, cfg.unlock_layers)
        self.comp = Compressor(k=cfg.k, msg_dim=cfg.msg_dim).to(device)
        self.dec = FactsDecoder(k=cfg.k, msg_dim=cfg.msg_dim).to(device)
        groups = [{"params": list(self.comp.parameters()) + list(self.dec.parameters()), "lr": cfg.lr}]
        if cfg.unlock_layers:
            groups.append({"params": self.reader.trainable(), "lr": cfg.reader_lr})
        self.base_lrs = [g["lr"] for g in groups]
        self.opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=1e-4)
        self.ds = OverheadQuestions(root / "train", train=True, seed=cfg.seed)
        self.val = {s: OverheadQuestions(root / s, train=False, seed=123) for s in ["val_rooms", "val_drive"] if (root / s).exists()}
        self.loader = torch.utils.data.DataLoader(self.ds, batch_size=cfg.batch, shuffle=True, num_workers=4, drop_last=True,
                                                  collate_fn=_collate, persistent_workers=True)
        self.n_trainable = {"reader": sum(p.numel() for p in self.reader.trainable()),
                            "compressor+decoder": sum(p.numel() for g in groups[:1] for p in g["params"])}
        self.step, self.log = 0, []

    def lr_factor(self, s):
        c = self.cfg
        return s / c.warmup if s < c.warmup else 0.5 * (1 + math.cos(math.pi * (s - c.warmup) / max(1, c.steps - c.warmup))) * 0.95 + 0.05

    def forward(self, feats, prompts):
        tokens, mask = self.reader(feats.to(self.dev), prompts)
        msg, _ = self.comp(tokens, mask)
        return msg, self.dec(msg)

    @staticmethod
    def loss(out, target, cls):
        reg = torch.cat([out[k] for k in FACTS if k != "dir4"], 1)
        mse = nn.functional.mse_loss(reg, target)
        ce = nn.functional.cross_entropy(out["dir4"], cls)
        return mse + 0.5 * ce, mse, ce

    def train(self, steps: int | None = None):
        """Generator: yields one log row every ``log_every`` steps (with val scores every ``val_every``)."""
        steps = steps or self.cfg.steps
        self.reader.train(); self.comp.train(); self.dec.train()
        it, t0 = iter(self.loader), time.time()
        while self.step < steps:
            try:
                feats, prompts, target, cls = next(it)
            except StopIteration:
                it = iter(self.loader); continue
            for g, lr in zip(self.opt.param_groups, self.base_lrs):
                g["lr"] = lr * self.lr_factor(self.step)
            _msg, out = self.forward(feats, prompts)
            loss, mse, ce = self.loss(out, target.to(self.dev), cls.to(self.dev))
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in self.opt.param_groups for p in g["params"]], 1.0)
            self.opt.step()
            self.step += 1
            if self.step % self.cfg.log_every == 0 or self.step == steps:
                row = {"step": self.step, "loss": loss.item(), "mse": mse.item(), "ce": ce.item(), "sec": time.time() - t0}
                if self.step % self.cfg.val_every == 0 or self.step == steps:
                    for name, ds in self.val.items():
                        row.update({f"{name}/{k}": v for k, v in self.evaluate(ds).items()})
                    self.reader.train(); self.comp.train(); self.dec.train()
                self.log.append(row)
                yield row

    @torch.no_grad()
    def evaluate(self, ds, batch: int = 128) -> dict:
        """The visibility-check scores, on the server's own predictions (decoded facts)."""
        self.reader.eval(); self.comp.eval(); self.dec.eval()
        outs, tgts, clss = [], [], []
        for s in range(0, len(ds), batch):
            items = [ds[i] for i in range(s, min(len(ds), s + batch))]
            f, p, t, c = _collate(items)
            _m, o = self.forward(f, p)
            outs.append({k: v.cpu() for k, v in o.items()}); tgts.append(t); clss.append(c)
        o = {k: torch.cat([x[k] for x in outs]) for k in FACTS}
        t, c = torch.cat(tgts), torch.cat(clss)

        def ang(pred, true):
            e = torch.rad2deg(torch.atan2(pred[:, 1], pred[:, 0]) - torch.atan2(true[:, 1], true[:, 0]))
            return ((e + 180) % 360 - 180).abs()
        return {
            "car_xy_m": float(((o["car_xy"] - t[:, 0:2]) * XY_SCALE).norm(dim=1).median()),
            "heading_deg": float(ang(o["heading"], t[:, 2:4]).median()),
            "obj_xy_m": float(((o["obj_xy"] - t[:, 4:6]) * XY_SCALE).norm(dim=1).median()),
            "rel_dir_deg": float(ang(o["rel_dir"], t[:, 6:8]).median()),
            "rel_dist_m": float(((o["rel_dist"] - t[:, 8:9]) * XY_SCALE).abs().median()),
            "dir4_acc": float((o["dir4"].argmax(1) == c).float().mean()),
            "dir4_majority": float(torch.bincount(c, minlength=4).max() / len(c)),
        }

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg.__dict__,
                    "reader_layers": {f"{i}": L.state_dict() for i, L in enumerate(self.reader.layers)},
                    "compressor": self.comp.state_dict(), "decoder": self.dec.state_dict(), "log": self.log}, path)
        return path


def plot_server_training(log, fig=None):
    """Loss and validation scores over training (the notebook's progress chart)."""
    import matplotlib.pyplot as plt
    fig = fig or plt.figure(figsize=(13, 3.6))
    axs = fig.subplots(1, 3)
    st = [r["step"] for r in log]
    axs[0].plot(st, [r["mse"] for r in log], label="position/direction error (MSE)")
    axs[0].plot(st, [r["ce"] for r in log], label="ahead/left/behind/right (cross-entropy)")
    axs[0].set_yscale("log"); axs[0].set_title("training loss"); axs[0].legend(fontsize=8)
    v = [r for r in log if any(k.endswith("dir4_acc") for k in r)]
    for split, ls in [("val_rooms", "-"), ("val_drive", "--")]:
        if v and f"{split}/dir4_acc" in v[0]:
            vs = [r["step"] for r in v]
            axs[1].plot(vs, [r[f"{split}/dir4_acc"] for r in v], ls, label=f"{split}")
            axs[2].plot(vs, [r[f"{split}/obj_xy_m"] for r in v], ls, label=f"{split}: named object")
            axs[2].plot(vs, [r[f"{split}/car_xy_m"] for r in v], ls, label=f"{split}: car")
    axs[1].axhline(0.8, color="grey", lw=0.8, ls=":"); axs[1].set_ylim(0, 1)
    axs[1].set_title("named object ahead/left/behind/right\n(val accuracy; pass mark 0.8)"); axs[1].legend(fontsize=8)
    axs[2].set_title("position error (val, metres, median)"); axs[2].legend(fontsize=7)
    for a in axs:
        a.set_xlabel("step")
    fig.tight_layout()
    return fig
