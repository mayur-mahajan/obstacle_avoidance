"""Guidance tokens for the car (notebook 06): the overhead server's help, slotted into the car's reader input.

    [ marker | front camera 64 tokens | marker | GUIDANCE TOKENS | instruction words | speed ]  -> car action head

The instruction comes after the guidance, so (with the reader's one-way attention) the words can relate the named object
to both what the car sees and what the server says. Kinds of guidance (one per model):
  facts    9 numbers, the simulator's TRUE facts (C*, upper bound) or the server's PREDICTED facts (C)
           car x, y, heading cos/sin, goal x, y, goal direction from the car cos/sin, distance  -> 8 tokens
  message  the server's 8 x 128 message (A, the main design)                                -> 8 tokens
  image    the overhead picture's whole view, 64 tokens (I, "just add a second camera")    -> 64 tokens
When the help is absent (dropped in training ~50% of the time, or switched off in tests) the guidance tokens are a
LEARNED "no help" set: the car always knows whether help is present.
"""

from __future__ import annotations

import math

import numpy as np

import torch
from torch import nn

DIM = 960            # the reader's width
N_TOKENS = {"facts": 8, "message": 8, "image": 64}
IN_DIM = {"facts": 9, "message": 128, "image": DIM}


class GuidanceAdapter(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        assert kind in N_TOKENS, kind
        self.kind, n = kind, N_TOKENS[kind]
        self.absent = nn.Parameter(torch.randn(n, DIM) * 0.02)
        if kind == "facts":
            self.net = nn.Sequential(nn.Linear(9, 512), nn.GELU(), nn.Linear(512, n * DIM))
        elif kind == "message":
            self.net = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, DIM))
            self.pos = nn.Parameter(torch.randn(n, DIM) * 0.02)
        self.gain = nn.Parameter(torch.ones(1))

    def forward(self, x, present):
        """x: facts (B, 9) / message (B, 8, 128) / image tokens (B, 64, 960); present: (B,) bool -> (B, n, 960)."""
        B = len(present)
        if self.kind == "facts":
            tok = self.net(x.float()).view(B, -1, DIM)
        elif self.kind == "message":
            tok = self.net(x.float()) + self.pos
        else:
            tok = x.float()                                   # SmolVLM2's own picture tokens: already the right scale
        if self.kind != "image":                               # new tokens: unit-size vectors times a learned gain
            tok = tok / tok.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * self.gain
        return torch.where(present.view(B, 1, 1), tok, self.absent.expand(B, -1, -1).to(tok.dtype))


def install(policy, adapter: GuidanceAdapter):
    """Wrap the car's prefix builder so the current guidance (policy.model._guidance, set per batch) is inserted right
    after the front camera's image block. No guidance set -> the learned "no help" tokens."""
    m = policy.model
    assert getattr(m, "_format_fixed", False) and m.add_image_special_tokens, "apply the reader fix first"
    # object.__setattr__: keep the adapter OUT of the car model's parameters (it is saved separately, *_guidance.pt);
    # a plain attribute would register it as a sub-module and its weights would end up in the car's checkpoint
    if getattr(m, "_guidance_installed", False):
        object.__setattr__(m, "_guidance_adapter", adapter)
        return policy
    orig = m.embed_prefix
    m._guidance = None
    object.__setattr__(m, "_guidance_adapter", adapter)

    def embed_prefix(*a, **k):
        embs, pad, att = orig(*a, **k)
        B = embs.shape[0]
        g = m._guidance if m._guidance is not None else m._guidance_adapter.absent.expand(B, -1, -1)
        g = g.to(embs.dtype)
        at = 1 + 64 + 1                                        # marker + front camera tokens + marker
        embs = torch.cat([embs[:, :at], g, embs[:, at:]], 1)
        pad = torch.cat([pad[:, :at], torch.ones(B, g.shape[1], dtype=pad.dtype, device=pad.device), pad[:, at:]], 1)
        att = torch.cat([att[:, :at], torch.ones(B, g.shape[1], dtype=att.dtype, device=att.device), att[:, at:]], 1)
        return embs, pad, att

    m.embed_prefix = embed_prefix
    m._guidance_installed = True
    return policy


class GuidanceStore:
    """Per-frame guidance inputs from scripts/overhead_frames.py, looked up by the frame's global index."""

    def __init__(self, root, kind: str, source: str = "true", n_frames: int | None = None, seed: int = 0,
                 route_file: str = "route_tracked.npy"):
        import numpy as np
        from pathlib import Path
        root = Path(root)
        meta = np.load(root / "meta.npz", allow_pickle=True)
        self.row_of = {int(g): i for i, g in enumerate(meta["row"])}
        self.room = meta["seed"]                                     # room (layout) of every row
        if kind == "facts":
            true = np.load(root / "facts_true.npy")[:, :9].astype(np.float32)
            pred = np.load(root / "facts_pred.npy")[:, :9].astype(np.float32)
            if source == "route":                                    # perfect route points (scripts/route_frames.py)
                true = np.load(root / "route_true.npy").astype(np.float32)
            if source == "route_real":                               # the real server's route (scripts/server_route_frames.py
                true = np.load(root / route_file).astype(np.float32)            # --tracker): tracked car + learned map
            self.x = {"true": true, "pred": pred, "noisy": true, "route": true, "route_real": true}[source]
            if source == "noisy":
                self.noise = ServerLikeNoise(true, pred, self.room, seed)
        elif kind == "message":
            self.x = np.load(root / "message.npy")
        else:
            self.x = np.load(root / "feats.npy", mmap_mode="r")      # (N, 5, 64, 960): index 4 = the whole view
        self.kind, self.source = kind, source

    def get(self, frames):
        import numpy as np
        rows = [self.row_of[int(f)] for f in frames]
        x = self.x[rows] if self.kind != "image" else np.stack([np.asarray(self.x[r, 4]) for r in rows])
        if self.kind == "facts" and self.source == "noisy":
            x = self.noise(np.asarray(rows))
        return torch.from_numpy(np.asarray(x, dtype=np.float32))


def _angle(c, s):
    return np.arctan2(s, c)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ServerLikeNoise:
    """True facts + errors resampled from the REAL server (option "noisy", notebook 06 step 1).

    The server's real mistakes on the driving frames (prediction - truth, all 9 numbers of one frame together, so their
    joint pattern is kept: e.g. a backwards heading reading and the wrong goal direction that comes with it) are pasted
    onto the true facts of another frame. The mistake always comes from a frame of a DIFFERENT room, so the car can't
    learn "this room's facts are wrong in this particular way". Per sample, how bad the help is (Miki et al. 2022):
        60%  normal     the server's error as measured
        30%  worse      the same error x 1.5 (backwards readings, > 135 deg, stay backwards)
        10%  useless    the true facts of a frame from another room
    Angles (heading, goal direction) get their error as a rotation; positions and distance get it added."""

    LEVELS = (0.6, 0.3, 0.1)

    def __init__(self, true, pred, room, seed: int = 0):
        self.true, self.room = true, np.asarray(room)
        self.rng = np.random.default_rng(seed)
        self.d_xy = pred[:, [0, 1, 4, 5]] - true[:, [0, 1, 4, 5]]                         # car xy, goal xy
        self.d_head = _wrap(_angle(pred[:, 2], pred[:, 3]) - _angle(true[:, 2], true[:, 3]))
        self.d_dir = _wrap(_angle(pred[:, 6], pred[:, 7]) - _angle(true[:, 6], true[:, 7]))
        self.d_dist = pred[:, 8] - true[:, 8]

    def _other_room(self, rows):
        """A random row from a different room for each row."""
        n, out = len(self.true), self.rng.integers(len(self.true), size=len(rows))
        same = self.room[out] == self.room[rows]
        while same.any():
            out[same] = self.rng.integers(n, size=same.sum())
            same = self.room[out] == self.room[rows]
        return out

    def __call__(self, rows):
        """Noisy facts for training rows (errors and useless facts taken from other rooms)."""
        rows = np.asarray(rows)
        level = self.rng.choice(3, size=len(rows), p=self.LEVELS)
        return self.perturb(self.true[rows], level, self._other_room(rows))

    def perturb(self, t, level, src=None):
        """Add real server errors to true facts t (n, 9). level per row: 0 normal, 1 x1.5, 2 useless.
        src: the rows whose errors are used (default: random rows of the pool)."""
        t = np.array(t, np.float32, copy=True)
        level = np.broadcast_to(np.asarray(level), (len(t),))
        if src is None:
            src = self.rng.integers(len(self.true), size=len(t))
        k = np.where(level == 1, 1.5, 1.0)[:, None]

        def scaled(d):                                   # angles: x1.5, but a backwards reading stays backwards
            return np.where(np.abs(d) > np.radians(135), d, k[:, 0] * d)

        t[:, [0, 1, 4, 5]] += k * self.d_xy[src]
        for (c, s), d in (((2, 3), self.d_head[src]), ((6, 7), self.d_dir[src])):
            a = _angle(t[:, c], t[:, s]) + scaled(d)
            t[:, c], t[:, s] = np.cos(a), np.sin(a)
        t[:, 8] = np.maximum(0.0, t[:, 8] + k[:, 0] * self.d_dist[src])
        useless = level == 2
        if useless.any():                                # the facts of some other place
            t[useless] = self.true[self.rng.integers(len(self.true), size=int(useless.sum()))]
        return t
