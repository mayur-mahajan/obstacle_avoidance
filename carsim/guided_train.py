"""Notebook 06: imitation training of the car WITH overhead guidance (carsim/guidance.py).

Same recipe as notebook 04 (carsim/bc_train.py: car head, SmolVLM2's input format, last 4 language layers nudged,
random-object quiz, 5000 steps, batch 32) plus:
  * guidance tokens after the front camera: facts (C* true / C predicted), the server's message (A) or the overhead
    whole view (I); kind "none" = front camera only (F)
  * help dropout: each training sample has the help removed with probability drop_p (decided: 0.5); the car then sees
    the learned "no help" tokens - so it must drive both with and without help, and always knows which
  * warm start (decided): A, C and I start from the trained C* car (all its trained weights), with a fresh adapter
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .bc_train import BCConfig, BCTrainer
from .device import autocast
from .guidance import GuidanceAdapter, GuidanceStore, install


@dataclass
class GuidedConfig(BCConfig):
    head: str = "car"
    quiz: str = "random"
    unlock_layers: int = 4
    guidance: str = "facts"          # facts | message | image | none
    source: str = "true"             # facts only: "true" (C*) or "pred" (C, the server's predicted facts)
    drop_p: float = 0.5              # chance a training sample has the help removed
    adapter_lr: float = 1e-4
    store: str = "runs/overhead_frames/route_follow"
    init: str | None = None          # warm start: a trained car checkpoint (e.g. C*)
    route_file: str = "route_tracked.npy"   # source "route_real": which real-route file (another map -> another file)
    val_source: str | None = None    # facts only: help used for the validation loss (default: same as source).
                                     # "noisy" training is validated on the REAL server's facts ("pred")


class GuidedBCTrainer(BCTrainer):
    def __init__(self, cfg: GuidedConfig, prep_path, device: str | None = None):
        super().__init__(cfg, prep_path, device)
        self.adapter = None
        if cfg.init:                                          # warm start: copy every trained weight of the earlier car
            sd = torch.load(cfg.init, map_location="cpu")
            named = dict(self.m.named_parameters())
            for n, p in sd.items():
                if n.startswith("_guidance_adapter."):         # (older checkpoints stored the adapter here too)
                    continue
                named[n].data.copy_(p.to(named[n].device, named[n].dtype))
        if cfg.guidance != "none":
            self.adapter = GuidanceAdapter(cfg.guidance).to(self.dev)
            install(self.policy, self.adapter)
            self.store = GuidanceStore(cfg.store, cfg.guidance, cfg.source, seed=cfg.seed, route_file=cfg.route_file)
            vs = cfg.val_source or cfg.source
            self.val_store = self.store if vs == cfg.source else GuidanceStore(cfg.store, cfg.guidance, vs, route_file=cfg.route_file)
            g = {"name": "guidance", "params": list(self.adapter.parameters()), "lr": cfg.adapter_lr}
            self.groups.append(g)
            self.opt.add_param_group({"params": g["params"], "lr": g["lr"]})
            self.base_lrs.append(g["lr"])
            self.n_trainable["guidance"] = sum(p.numel() for p in g["params"])
        self.grng = np.random.default_rng(cfg.seed + 5)

    def set_guidance(self, frames, present, store=None):
        """Guidance tokens for this batch (frames = global frame numbers; present = (B,) bool)."""
        if self.adapter is None:
            return
        x = (store or self.store).get(frames).to(self.dev)
        self.m._guidance = self.adapter(x, present.to(self.dev))

    def train(self, steps: int | None = None):
        """Notebook 04's loop with guidance set for every batch."""
        c, m = self.cfg, self.m
        stop = min(c.steps, self.step + steps) if steps else c.steps
        m.train(); self.vw.get_vlm_model().vision_model.eval()
        t0 = time.time(); acc = {"flow": [], "quiz": [], "hit": []}
        while self.step < stop:
            for b in self.loader:
                for g, lr0 in zip(self.opt.param_groups, self.base_lrs):
                    g["lr"] = lr0 * self.lr_factor(self.step)
                B = len(b["task"])
                present = torch.from_numpy(self.grng.random(B) >= c.drop_p)
                act = torch.zeros(B, 10, self.policy.config.max_action_dim, device=self.dev); act[:, :, :3] = b["action"].to(self.dev)
                with autocast(self.dev):
                    self.set_guidance(b["frame"], present)
                    inp = self.prep_in(b["image"], b["task"], b["state"])
                    flow = m.forward(*inp, act)[:, :, :3].mean()
                    loss = flow
                    if self.quiz is not None:
                        prompts, lab = self.quiz_questions(b, self.qrng)
                        lg = self.quiz_logits(b["image"], prompts, b["state"])
                        lab = lab.to(self.dev)
                        ql = F.cross_entropy(lg.float(), lab)
                        loss = flow + c.quiz_weight * ql
                        acc["quiz"].append(ql.item()); acc["hit"].append((lg.argmax(1) == lab).float().mean().item())
                self.m._guidance = None                               # the tokens are already in the graph
                self.opt.zero_grad(set_to_none=True); loss.backward()
                gn = torch.nn.utils.clip_grad_norm_([p for g in self.groups for p in g["params"]], 10.0)
                self.opt.step(); self.step += 1; acc["flow"].append(flow.item())
                if self.step % c.log_every == 0 or self.step == stop:
                    row = {"step": self.step, "driving_loss": float(np.mean(acc["flow"])), "grad_norm": float(gn),
                           "lr_factor": self.lr_factor(self.step), "minutes": (time.time() - t0) / 60}
                    if self.quiz is not None:
                        row.update(quiz_loss=float(np.mean(acc["quiz"])), quiz_train_acc=float(np.mean(acc["hit"])))
                    if self.step % c.val_every == 0 or self.step == stop:
                        row.update(self.val_driving_loss())
                    acc = {"flow": [], "quiz": [], "hit": []}
                    self.log.append(row)
                    yield row
                if self.step >= stop:
                    break

    @torch.no_grad()
    def val_driving_loss(self, n_batches: int = 20):
        """Driving loss on validation drives WITH help and WITHOUT help (same frames, same noise): if the car uses the
        help, the loss with help must be clearly lower."""
        if self.adapter is None:
            return {}
        self.m.eval()
        vl = torch.utils.data.DataLoader(self.val, batch_size=32, shuffle=False, num_workers=2,
                                         collate_fn=self.loader.collate_fn)
        out = {"val_loss_with_help": [], "val_loss_no_help": []}
        for i, b in enumerate(vl):
            if i >= n_batches:
                break
            B = len(b["task"])
            act = torch.zeros(B, 10, self.policy.config.max_action_dim, device=self.dev); act[:, :, :3] = b["action"].to(self.dev)
            g = torch.Generator(device=self.dev).manual_seed(i)
            noise = torch.randn(act.shape, device=self.dev, generator=g)
            time_ = torch.rand(B, device=self.dev, generator=g) * 0.999 + 0.001
            for key, pres in [("val_loss_with_help", True), ("val_loss_no_help", False)]:
                with autocast(self.dev):
                    self.set_guidance(b["frame"], torch.full((B,), pres), self.val_store)
                    inp = self.prep_in(b["image"], b["task"], b["state"])
                    l = self.m.forward(*inp, act, noise=noise, time=time_)[:, :, :3].mean()
                out[key].append(l.item())
        self.m._guidance = None
        self.m.train(); self.vw.get_vlm_model().vision_model.eval()
        return {k: float(np.mean(v)) for k, v in out.items()}

    def save(self, path):
        from pathlib import Path
        p = super().save(path)
        if self.adapter is not None:
            torch.save({"kind": self.cfg.guidance, "source": self.cfg.source, "state": self.adapter.state_dict()},
                       Path(p).with_name(Path(p).stem + "_guidance.pt"))
        return p
