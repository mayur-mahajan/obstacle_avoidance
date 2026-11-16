"""The SERVER of notebook 06 built on the full, unmodified SmolVLM2-500M (all 32 language layers, its own input format and
image splitting), instead of smolVLA's cut-down copy (16 layers, one 64-token view, no tiling).

    overhead picture, 1024 px -> 4 tiles + whole view, 5 x 64 tokens (pre-computed: scripts/overhead_data_native.py)
    + instruction, in SmolVLM2's chat format:  "<|im_start|>User:<5 picture blocks><instruction><end_of_utterance>\\nAssistant:"
        -> SmolVLM2's 32 language layers (last N trainable) -> output tokens
        -> position-aware compressor (8 queries, optionally steered by the instruction's word tokens) -> MESSAGE 8 x 128
        -> facts decoder (training signal; option C)

Why the changes vs carsim/server.py (diagnosis 2026-11-14): smolVLA's 16-layer, 64-token reader could not tie the words
to the object (identity given directly: object found to 0.74 m; via the words: 2.0 m); the compressor ignored WHICH token
a feature came from (probes lose 30-50% without it); one 64-token view loses detail (tiling improved heading 33 -> 20 deg).
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
from .server import FACTS, XY_SCALE, FactsDecoder

SMOLVLM2_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TILES = 5


def load_smolvlm2(device="cuda"):
    """The full SmolVLM2-500M (bf16) and its processor, set to split a 1024 px picture into 2 x 2 tiles + the whole view."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(SMOLVLM2_ID)
    proc.image_processor.size = {"longest_edge": 1024}
    model = AutoModelForImageTextToText.from_pretrained(SMOLVLM2_ID, dtype=torch.bfloat16).to(device).eval()
    return model, proc


class NativeReader(nn.Module):
    """SmolVLM2's language model on (cached picture tokens, instruction) in SmolVLM2's own chat format."""

    def __init__(self, model, proc, unlock_layers: int = 4):
        super().__init__()
        from PIL import Image
        self.model, self.tok = model, proc.tokenizer
        self.lm = model.model.text_model
        for p in model.parameters():
            p.requires_grad = False
        self.layers = list(self.lm.layers[-unlock_layers:]) if unlock_layers else []
        for L in self.layers:
            for p in L.parameters():
                p.data = p.data.float(); p.requires_grad = True
        # the fixed part of the input: chat header + the 5 picture blocks (row/column markers), then the words, then the end
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "QQQ"}]}]
        ids = proc(text=proc.apply_chat_template(msgs, add_generation_prompt=True), images=[Image.new("RGB", (1024, 1024))],
                   return_tensors="pt")["input_ids"][0].tolist()
        q = self.tok("QQQ", add_special_tokens=False)["input_ids"]
        i = next(k for k in range(len(ids)) if ids[k:k + len(q)] == q)
        self.prefix, self.suffix = ids[:i], ids[i + len(q):]
        self.img_pos = torch.tensor([k for k, t in enumerate(self.prefix) if t == model.config.image_token_id])
        assert len(self.img_pos) == TILES * 64, len(self.img_pos)

    def trainable(self):
        return [p for L in self.layers for p in L.parameters()]

    def build(self, prompts, device):
        """-> input ids (B, L), attention mask, and a mask of the instruction's word tokens."""
        words = [self.tok(" " + p.strip(), add_special_tokens=False)["input_ids"] for p in prompts]
        seqs = [self.prefix + w + self.suffix for w in words]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), self.tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros(len(seqs), L, dtype=torch.bool)
        wm = torch.zeros(len(seqs), L, dtype=torch.bool)
        for b, (s, w) in enumerate(zip(seqs, words)):
            ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = True
            wm[b, len(self.prefix): len(self.prefix) + len(w)] = True
        return ids.to(device), att.to(device), wm.to(device)

    def forward(self, feats, prompts):
        """feats (B, 5, 64, 960) -> output tokens (B, L, 960) float, attention mask, word mask."""
        dev = feats.device
        ids, att, wm = self.build(prompts, dev)
        emb = self.lm.get_input_embeddings()(ids)
        emb[:, self.img_pos.to(dev)] = feats.flatten(1, 2).to(emb.dtype)          # what SmolVLM2's own forward does
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = self.lm(inputs_embeds=emb, attention_mask=att.long()).last_hidden_state
        return out.float(), att, wm


class PosCompressor(nn.Module):
    """Position-aware compressor: each token gets a learned embedding of its place in the sequence (= which tile and
    grid cell, or which word), 8 learned queries attend, optionally steered by the instruction ("words": the mean of
    the reader's outputs at the word tokens is added to every query, Q-Former style)."""

    def __init__(self, max_tok: int = 400, dim: int = 960, k: int = 8, d: int = 256, msg_dim: int = 128, steer: bool = True):
        super().__init__()
        self.kv = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d))
        self.pos = nn.Parameter(torch.randn(1, max_tok, d) * 0.02)
        self.query = nn.Parameter(torch.randn(1, k, d) / math.sqrt(d))
        self.steer = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, d)) if steer else None
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, msg_dim))

    def forward(self, tokens, mask, word_mask=None, ident=None):
        kv = self.kv(tokens) + self.pos[:, : tokens.shape[1]]
        q = self.query.expand(len(kv), -1, -1)
        if self.steer is not None and word_mask is not None:
            w = (tokens * word_mask[..., None]).sum(1) / word_mask.sum(1, keepdim=True).clamp(min=1)
            q = q + self.steer(w)[:, None]
        if ident is not None:                                   # diagnosis only: identity handed over directly
            q = q + ident[:, None]
        a, _ = self.attn(q, kv, kv, key_padding_mask=~mask)
        return self.out(a + self.mlp(a))


class NativeQuestions(torch.utils.data.Dataset):
    """Like carsim.server.OverheadQuestions, for (5, 64, 960) features; also returns the asked object's identity."""

    def __init__(self, root, train: bool, seed: int = 0):
        from .server import OverheadQuestions
        self.q = OverheadQuestions.__new__(OverheadQuestions)
        OverheadQuestions.__init__(self.q, root, train, seed)

    def __len__(self):
        return len(self.q)

    def __getitem__(self, i):
        q = self.q
        k, prompt = q._pick(i, np.random.default_rng()) if q.train else q.fixed[i]
        f = overhead.facts(q.pose[i], q.objects[i, k, 2:4])
        t = np.concatenate([f["car_xy"] / XY_SCALE, f["heading"], f["obj_xy"] / XY_SCALE, f["rel_dir"], f["rel_dist"] / XY_SCALE])
        ident = int(q.objects[i, k, 1]) * 3 + int(q.objects[i, k, 0])
        return (torch.from_numpy(np.array(q.feats[i])), prompt, torch.from_numpy(t.astype(np.float32)),
                int(overhead.direction_class(f["bearing"])), ident)


def _collate(items):
    f, p, t, c, d = zip(*items)
    return torch.stack(f), list(p), torch.stack(t), torch.tensor(c), torch.tensor(d)


@dataclass
class NativeServerConfig:
    steps: int = 6000
    batch: int = 32
    seed: int = 0
    unlock_layers: int = 4
    reader_lr: float = 3e-5
    lr: float = 3e-4
    warmup: int = 300
    steer: bool = True             # compressor queries steered by the instruction's word tokens
    identity: bool = False         # DIAGNOSIS: no words; the asked object's identity is given to the compressor directly
    train_splits: tuple = ("train",)   # e.g. ("train", "train_heading"): + arrow practice (each spot at 4 headings)
    # loss weights per fact (heading retraining, 2026-11-14: heading / car-relative direction weighted up)
    w_car_xy: float = 1.0
    w_heading: float = 1.0
    w_obj_xy: float = 1.0
    w_rel_dir: float = 1.0
    w_rel_dist: float = 1.0
    w_dir4: float = 0.5
    init: str | None = None        # warm start from a saved server checkpoint (same unlock_layers / steer)
    log_every: int = 50
    val_every: int = 1000


class NativeServerTrainer:
    def __init__(self, cfg: NativeServerConfig, data_root, device: str = "cuda"):
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.cfg, self.dev, root = cfg, torch.device(device), Path(data_root)
        model, proc = load_smolvlm2(device)
        self.reader = NativeReader(model, proc, cfg.unlock_layers)
        self.comp = PosCompressor(steer=cfg.steer and not cfg.identity).to(device)
        self.dec = FactsDecoder().to(device)
        self.ident = nn.Embedding(18, 256).to(device) if cfg.identity else None
        new = list(self.comp.parameters()) + list(self.dec.parameters()) + (list(self.ident.parameters()) if self.ident else [])
        groups = [{"params": new, "lr": cfg.lr}] + ([{"params": self.reader.trainable(), "lr": cfg.reader_lr}] if cfg.unlock_layers else [])
        self.base_lrs = [g["lr"] for g in groups]
        self.opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=1e-4)
        self.ds = torch.utils.data.ConcatDataset([NativeQuestions(root / s, train=True, seed=cfg.seed) for s in cfg.train_splits])
        self.val = {s: NativeQuestions(root / s, train=False, seed=123) for s in ["val_rooms", "val_drive", "val_heading"] if (root / s).exists()}
        if cfg.init:
            ck = torch.load(cfg.init, map_location="cpu", weights_only=False)
            for i, L in enumerate(self.reader.layers):
                L.load_state_dict(ck["reader_layers"][str(i)])
            self.comp.load_state_dict(ck["compressor"]); self.dec.load_state_dict(ck["decoder"])
        self.loader = torch.utils.data.DataLoader(self.ds, batch_size=cfg.batch, shuffle=True, num_workers=4, drop_last=True,
                                                  collate_fn=_collate, persistent_workers=True)
        self.n_trainable = {"reader": sum(p.numel() for p in self.reader.trainable()), "new": sum(p.numel() for p in new)}
        self.step, self.log = 0, []

    def lr_factor(self, s):
        c = self.cfg
        return s / c.warmup if s < c.warmup else 0.5 * (1 + math.cos(math.pi * (s - c.warmup) / max(1, c.steps - c.warmup))) * 0.95 + 0.05

    def forward(self, feats, prompts, ident):
        if self.cfg.identity:
            prompts = [""] * len(prompts)                       # the reader never sees which object is asked
        tok, att, wm = self.reader(feats.to(self.dev), prompts)
        msg = self.comp(tok, att, wm, self.ident(ident.to(self.dev)) if self.ident is not None else None)
        return msg, self.dec(msg)

    def loss(self, out, target, cls):
        """Weighted sum of the per-fact errors (target layout: car_xy 2, heading 2, obj_xy 2, rel_dir 2, rel_dist 1)."""
        c = self.cfg
        parts = [("car_xy", 0, 2, c.w_car_xy), ("heading", 2, 4, c.w_heading), ("obj_xy", 4, 6, c.w_obj_xy),
                 ("rel_dir", 6, 8, c.w_rel_dir), ("rel_dist", 8, 9, c.w_rel_dist)]
        mse = sum(w * nn.functional.mse_loss(out[k], target[:, a:b]) for k, a, b, w in parts) / sum(p[3] for p in parts)
        ce = nn.functional.cross_entropy(out["dir4"], cls)
        return mse * sum(p[3] for p in parts) / len(parts) + c.w_dir4 * ce, mse, ce

    def train(self, steps=None):
        steps = steps or self.cfg.steps
        it, t0 = iter(self.loader), time.time()
        while self.step < steps:
            try:
                f, p, t, c, d = next(it)
            except StopIteration:
                it = iter(self.loader); continue
            for g, lr in zip(self.opt.param_groups, self.base_lrs):
                g["lr"] = lr * self.lr_factor(self.step)
            _m, out = self.forward(f, p, d)
            loss, mse, ce = self.loss(out, t.to(self.dev), c.to(self.dev))
            self.opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([q for g in self.opt.param_groups for q in g["params"]], 1.0)
            self.opt.step(); self.step += 1
            if self.step % self.cfg.log_every == 0 or self.step == steps:
                row = {"step": self.step, "loss": loss.item(), "mse": mse.item(), "ce": ce.item(), "sec": time.time() - t0}
                if self.step % self.cfg.val_every == 0 or self.step == steps:
                    for name, ds in self.val.items():
                        row.update({f"{name}/{k}": v for k, v in self.evaluate(ds).items()})
                self.log.append(row)
                yield row

    @torch.no_grad()
    def evaluate(self, ds, batch: int = 64):
        self.comp.eval(); self.dec.eval()
        outs, tg, cl = [], [], []
        for s in range(0, len(ds), batch):
            f, p, t, c, d = _collate([ds[i] for i in range(s, min(len(ds), s + batch))])
            _m, o = self.forward(f, p, d)
            outs.append({k: v.cpu() for k, v in o.items()}); tg.append(t); cl.append(c)
        self.comp.train(); self.dec.train()
        o = {k: torch.cat([x[k] for x in outs]) for k in FACTS}; t, c = torch.cat(tg), torch.cat(cl)
        ang = lambda a, b: (((torch.rad2deg(torch.atan2(a[:, 1], a[:, 0]) - torch.atan2(b[:, 1], b[:, 0])) + 180) % 360) - 180).abs()
        return {"car_xy_m": float(((o["car_xy"] - t[:, 0:2]) * XY_SCALE).norm(dim=1).median()),
                "heading_deg": float(ang(o["heading"], t[:, 2:4]).median()),
                "obj_xy_m": float(((o["obj_xy"] - t[:, 4:6]) * XY_SCALE).norm(dim=1).median()),
                "rel_dir_deg": float(ang(o["rel_dir"], t[:, 6:8]).median()),
                "rel_dir_within45": float((ang(o["rel_dir"], t[:, 6:8]) < 45).float().mean()),   # "points roughly the right way"
                "heading_reversed": float((ang(o["heading"], t[:, 2:4]) > 120).float().mean()),   # arrow read backwards
                "rel_dist_m": float(((o["rel_dist"] - t[:, 8:9]) * XY_SCALE).abs().median()),
                "dir4_acc": float((o["dir4"].argmax(1) == c).float().mean()),
                # the same 4-way answer read off the predicted direction ANGLE instead of the separate 4-way output
                "dir4_from_angle_acc": float((torch.from_numpy(overhead.direction_class(
                    torch.atan2(o["rel_dir"][:, 1], o["rel_dir"][:, 0]).numpy())) == c).float().mean()),
                "dir4_majority": float(torch.bincount(c, minlength=4).max() / len(c))}

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg.__dict__, "reader_layers": {str(i): L.state_dict() for i, L in enumerate(self.reader.layers)},
                    "compressor": self.comp.state_dict(), "decoder": self.dec.state_dict(),
                    "ident": self.ident.state_dict() if self.ident else None, "log": self.log}, path)
        return path


class ServerRuntime:
    """The trained server at test time: overhead picture (1024 px) + instruction -> message, decoded facts, and the
    picture's whole-view tokens (for the "full overhead picture" car)."""

    def __init__(self, ckpt, device: str = "cuda"):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = NativeServerConfig(**{k: v for k, v in ck["cfg"].items() if k in NativeServerConfig.__dataclass_fields__})
        self.model, self.proc = load_smolvlm2(device)
        self.reader = NativeReader(self.model, self.proc, cfg.unlock_layers)
        for i, L in enumerate(self.reader.layers):
            L.load_state_dict(ck["reader_layers"][str(i)])
        self.comp = PosCompressor(steer=cfg.steer).to(device); self.comp.load_state_dict(ck["compressor"])
        self.dec = FactsDecoder().to(device); self.dec.load_state_dict(ck["decoder"])
        self.comp.eval(); self.dec.eval(); self.dev = device

    @torch.no_grad()
    def __call__(self, pics, prompts):
        from PIL import Image
        pv = self.proc.image_processor([[Image.fromarray(p)] for p in pics], return_tensors="pt")
        f = self.model.model.get_image_features(pv["pixel_values"].to(self.dev, torch.bfloat16), pv["pixel_attention_mask"].to(self.dev),
                                                return_dict=True).pooler_output
        f = (torch.cat(list(f), 0) if isinstance(f, (list, tuple)) else f).reshape(len(pics), TILES, 64, -1).float()
        tok, att, wm = self.reader(f, list(prompts))
        msg = self.comp(tok, att, wm)
        out = self.dec(msg)
        return {"message": msg.float(), "facts_pred": torch.cat([out[k] for k in FACTS], 1).float(), "global_tokens": f[:, 4], "feats": f}
