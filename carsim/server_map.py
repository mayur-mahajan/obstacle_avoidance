"""The server's obstacle map (notebook 06, "real route" step): from the overhead picture, mark every 10 cm floor cell
of the room as blocked (box, wall segment, cylinder, coloured object, room wall) or free.

Why a learned map: obstacles and three of the four floors are all greys, so colour rules can't separate them; the
server has to learn what an obstacle looks like from above (user decision 2026-11-15). Why a separate geometric
planner afterwards: language/vision models plan paths badly (Mobility VLA 2407.07775: VLM waypoints 0% success vs
VLM-picks-goal + graph planner 90%; CoFL 2603.02854: VLM planners collide 80-97%), so the model only draws the map.

Input: the server's picture features, exactly as SmolVLM2 makes them (4 tiles + whole view = 5 x 64 tokens of 960;
cached by scripts/overhead_data_native.py / overhead_frames.py; frozen). The words are not needed: the map is the
same whatever the instruction.

    tokens (320 x 960) + "which tile, which spot" embedding -> 256 wide -> 2 transformer layers
    400 floor queries (a 20 x 20 grid of 40 cm cells, each knowing its floor position) -> 3 cross-attention layers
    -> each query paints its 4 x 4 block of 10 cm cells -> 80 x 80 logits -> 3 small conv layers to smooth edges
Labels: the simulator's occupancy (carsim/world.py occupancy, 5 cm grid) shrunk to 10 cm: a 10 cm cell is blocked if
any part of it is (missing an obstacle is worse than a slightly fat one).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_MAP = 80            # 80 x 80 cells of 10 cm = the 8 m room
Q = 20                # 20 x 20 queries, each paints 4 x 4 cells
D = 256


def label_map(spec, grid) -> np.ndarray:
    """(80, 80) bool: blocked 10 cm cells (index [i, j] = (x, y), as carsim.world.Grid)."""
    from .world import occupancy
    occ = occupancy(grid, spec.objects, spec.obstacles)                    # 160 x 160 at 5 cm
    return occ.reshape(N_MAP, 2, N_MAP, 2).any((1, 3))


def _sincos(pos, dim):
    """2-D sine/cosine position code for (n, 2) positions in [-1, 1]."""
    f = torch.exp(torch.arange(dim // 4, dtype=torch.float32) * (-math.log(100.0) / (dim // 4)))
    a = pos[:, :1] * f * math.pi, pos[:, 1:] * f * math.pi
    return torch.cat([a[0].sin(), a[0].cos(), a[1].sin(), a[1].cos()], 1)


class MapHead(nn.Module):
    def __init__(self, in_dim: int = 960):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, D))
        self.tok_pos = nn.Parameter(torch.randn(5 * 64, D) * 0.02)          # which tile + which spot in it
        enc = nn.TransformerEncoderLayer(D, 8, 4 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, 2, enable_nested_tensor=False)
        c = (torch.arange(Q, dtype=torch.float32) + 0.5) / Q * 2 - 1
        gx, gy = torch.meshgrid(c, c, indexing="ij")
        self.register_buffer("q_code", _sincos(torch.stack([gx.ravel(), gy.ravel()], 1), D))
        self.q = nn.Parameter(torch.zeros(Q * Q, D))
        dec = nn.TransformerDecoderLayer(D, 8, 4 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec, 3)
        self.paint = nn.Linear(D, 16 * 16)                                  # 4 x 4 cells x 16 features
        self.refine = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, feats):
        """feats (B, 5, 64, 960) -> logits (B, 80, 80); index [i, j] = cell (x, y)."""
        B = feats.shape[0]
        t = self.inp(feats.float().reshape(B, 320, -1)) + self.tok_pos
        mem = self.enc(t)
        q = (self.q + self.q_code).expand(B, -1, -1)
        h = self.dec(q, mem)                                                # (B, 400, D)
        p = self.paint(h).view(B, Q, Q, 4, 4, 16).permute(0, 5, 1, 3, 2, 4).reshape(B, 16, N_MAP, N_MAP)
        return self.refine(p)[:, 0]


def map_loss(logits, target, pos_weight: float = 5.0):
    """Scored on the room's INSIDE only (the outer walls are always the same; scoring them let the model get a decent
    loss by drawing only the walls). Cross-entropy that counts a missed blocked cell 5x (a missed box = a crash;
    ~5% of inside cells are blocked) + a soft-overlap (Dice) term."""
    t = target.float()[:, 2:-2, 2:-2]
    lg = logits[:, 2:-2, 2:-2]
    bce = F.binary_cross_entropy_with_logits(lg, t, pos_weight=torch.tensor(pos_weight, device=logits.device))
    p = lg.sigmoid()
    dice = 1 - (2 * (p * t).sum((1, 2)) + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def map_scores(logits, target, thr: float = 0.5):
    """Cell scores, ignoring the room's outer walls (always known): overlap (IoU), share of blocked cells found
    (recall), share of predicted-blocked cells that are really blocked (precision)."""
    inner = torch.zeros_like(target, dtype=torch.bool); inner[:, 2:-2, 2:-2] = True
    p, t = (logits.sigmoid() > thr) & inner, target.bool() & inner
    tp = (p & t).sum().item(); fp = (p & ~t).sum().item(); fn = (~p & t).sum().item()
    return {"iou": tp / max(1, tp + fp + fn), "recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp)}
