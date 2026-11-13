"""Imitation (behaviour-cloning) training of smolVLA on the scripted demos (notebook 04).

The recipe:
  * reader input format  SmolVLM2's own (image markers, one-way attention, no x31 input scaling; carsim.vla)
  * unlocked reader      the last N language layers of the reader may change (small learning rate); the vision part
                         and the earlier layers stay locked.
  * random-object quiz   at every training frame a small extra head answers "where is the <colour> <shape>?"
                         (not visible / left / centre / right) about a random object of the scene, mostly one in view.
                         The simulator recorded the answer, so it is graded automatically. Only used in training.
  * action head          "smolvla" = smolVLA's arm-pretrained action expert, or "car" = the same design, fresh, 3 actions

Usage (notebook 04):
    tr = BCTrainer(BCConfig(head="car"), prep_path)
    for row in tr.train():          # yields a log row every `log_every` steps
        ...plot...
    tr.save("runs/checkpoints/name.pt")   # loads with carsim.vla.load_bc_policy

Training is deterministic: the same settings, data and seed give the same weights (on the same kind of GPU).
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .device import autocast

CLASSES = ["not visible", "left", "centre", "right"]
CENTRE_DEG = 10.0                 # |bearing| below this = "centre"
HELD_OUT = ("orange", "sphere")   # held-out test goal: never named in training


@dataclass
class BCConfig:
    steps: int = 5000             # training steps (batches); ~35 min on an RTX 4080 SUPER
    batch: int = 32
    seed: int = 0
    lr: float = 1e-4              # driver (action expert) learning rate
    warmup: int = 300             # learning rate ramps up over these steps, then decays (cosine) to 5%
    unlock_layers: int = 4        # last N language layers of the reader that may change (0 = reader locked)
    reader_lr: float = 1e-5       # 10x smaller than the driver's: nudge the reader, don't overwrite it
    quiz: str = "random"          # "random" object of the scene; "goal" (only the goal: learns the shortcut "the obvious
                                  #   object"), or "none"
    quiz_weight: float = 0.1      # how much the quiz counts compared with the driving loss
    quiz_lr: float = 1e-3         # the quiz head is small and new, so it learns fast
    quiz_p_visible: float = 0.8   # chance the quiz asks about an object IN VIEW (where reading matters)
    mirror_p: float = 0.0         # chance a training sample is mirrored left<->right (0 = off: the scripted driver always
                                  #   searches left, and mirroring would make that habit a coin flip; notebook 03)
    head: str = "smolvla"         # "smolvla" = its arm-pretrained action expert; "car" = fresh expert, same design, 3-number actions
    fmt_markers: bool = True      # reader input format (carsim.vla.set_input_format), SmolVLM2's own: image marker tokens,
    fmt_causal: bool = True       #   one-way attention over picture + words,
    fmt_unscaled: bool = True     #   and no x31 scaling of the input pieces
    log_every: int = 100
    val_every: int = 1000


def quiz_labels_from_aux(aux):
    """Goal-only quiz labels from aux = (sin bearing, cos bearing, surface distance, visible). Bearing + = left."""
    b = torch.rad2deg(torch.atan2(aux[:, 0], aux[:, 1]))
    lab = torch.where(b > CENTRE_DEG, 1, torch.where(b < -CENTRE_DEG, 3, 2))
    return torch.where(aux[:, 3] > 0.5, lab, torch.zeros_like(lab))


def build_frame_objects(table):
    """{global frame index: [(k, colour, shape, bearing_rad, visible)]}, rebuilt from the episode seed + recorded pose."""
    from .vla import objects_in_view
    from .world import EnvConfig, Grid, sample_episode
    cfg = EnvConfig(); grid = Grid(cfg); specs = {}; out = {}
    for idx, seed, x, y, yaw in table[["index", "sim.episode_seed", "sim.pose.x", "sim.pose.y", "sim.pose.yaw"]].itertuples(index=False):
        seed = int(seed)
        if seed not in specs:
            specs[seed] = sample_episode(seed, cfg)
        out[int(idx)] = objects_in_view(specs[seed], (x, y, yaw), cfg, grid)
    return out


def _collate(items):
    return {k: (torch.stack([s[k] for s in items]) if k in ("image", "state", "action", "aux") else [s[k] for s in items])
            for k in ("image", "state", "action", "aux", "task", "frame", "mirrored")}


class BCTrainer:
    def __init__(self, cfg: BCConfig, prep_path, device: str | None = None):
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        from .data import ActionScaler, NavChunkDataset, Paraphraser, load_training_data
        from .device import pick_device
        from .vla import load_smolvla, make_car_head, set_input_format

        device = device or pick_device()
        self.cfg, self.dev, self.prep_path = cfg, device, str(prep_path)
        self._make_att_2d_masks = make_att_2d_masks
        torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

        # ---- model: smolVLA, its reader fed in SmolVLM2's own format ----
        self.policy = load_smolvla(device)
        set_input_format(self.policy, markers=cfg.fmt_markers, causal=cfg.fmt_causal, unscaled=cfg.fmt_unscaled)
        if cfg.head == "car":
            make_car_head(self.policy, seed=cfg.seed)
        self.m, self.vw = self.policy.model, self.policy.model.vlm_with_expert
        self.policy.config.chunk_size = 10

        # ---- data ----
        prep, table, frames, eps = load_training_data(prep_path)
        self.prep, self.table = prep, table
        sc = ActionScaler(np.array(prep["action_limits"], np.float32))
        self.ds = NavChunkDataset(table, frames, eps, prep["split"]["train"], sc, chunk=10, mirror_p=cfg.mirror_p, paraphrase=True)
        self.val = NavChunkDataset(table, frames, eps, prep["split"]["val"], sc, chunk=10, mirror_p=0.0, paraphrase=False)
        self.frame_objects = build_frame_objects(table) if cfg.quiz == "random" else None
        self.qpara = Paraphraser("train", seed=cfg.seed + 7)
        self.data_note = f"{len(self.ds)} frames, plain shuffling"
        self.loader = torch.utils.data.DataLoader(self.ds, batch_size=cfg.batch, num_workers=4, collate_fn=_collate,
                                                  worker_init_fn=self.ds.worker_init, persistent_workers=True, drop_last=True,
                                                  shuffle=True)

        # ---- what learns ----
        m, vw = self.m, self.vw
        for p in m.parameters():
            p.requires_grad = False
        driver = [vw.lm_expert, m.state_proj, m.action_in_proj, m.action_out_proj, m.action_time_mlp_in, m.action_time_mlp_out]
        groups = [{"name": "driver", "params": [p for mod in driver for p in mod.parameters()], "lr": cfg.lr}]
        if cfg.unlock_layers:
            layers = vw.get_vlm_model().text_model.layers[-cfg.unlock_layers:]
            groups.append({"name": "reader", "params": [p for L in layers for p in L.parameters()], "lr": cfg.reader_lr})
        self.quiz = None
        if cfg.quiz != "none":
            hid = vw.lm_expert.config.hidden_size
            self.quiz = torch.nn.Sequential(torch.nn.Linear(hid, 256), torch.nn.GELU(), torch.nn.Linear(256, len(CLASSES))).to(device)
            groups.append({"name": "quiz", "params": list(self.quiz.parameters()), "lr": cfg.quiz_lr})
        for g in groups:
            for p in g["params"]:
                p.data = p.data.float(); p.requires_grad = True   # fp32 master copies (bf16 compute on NVIDIA GPUs)
        self.groups = groups
        self.n_trainable = {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}
        self.opt = torch.optim.AdamW([{"params": g["params"], "lr": g["lr"]} for g in groups], betas=(0.9, 0.95), weight_decay=1e-10)
        self.base_lrs = [g["lr"] for g in groups]
        self.step, self.log = 0, []
        self.qrng = np.random.default_rng(cfg.seed + 99)

    # ---------------------------------------------------------------------------------------------
    def lr_factor(self, s):
        c = self.cfg
        warm = min(c.warmup, c.steps // 10)
        return s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, c.steps - warm))) * 0.95 + 0.05

    def prep_in(self, images, tasks, states):
        from .vla import format_prompts, prepare_inputs
        return prepare_inputs(self.policy, images, format_prompts(self.policy, tasks), states, device=self.dev)

    def expert_out(self, inp, x_t, t):
        """smolVLA's forward, returning the expert's output at the action tokens (same code path as model.forward)."""
        m, vw = self.m, self.vw
        pe, pp, pa = m.embed_prefix(*inp[:4], state=inp[4])
        se, sp, sa = m.embed_suffix(x_t, t)
        pad, att = torch.cat([pp, sp], 1), torch.cat([pa, sa], 1)
        (_, so), _ = vw.forward(attention_mask=self._make_att_2d_masks(pad, att), position_ids=torch.cumsum(pad, 1) - 1,
                                past_key_values=None, inputs_embeds=[pe, se], use_cache=False)
        return so[:, -self.policy.config.chunk_size:].float()

    def quiz_questions(self, b, rng):
        """-> (prompts, labels). random: name a random object of the scene (80% one in view); goal: the driving prompt."""
        if self.cfg.quiz == "goal":
            return b["task"], quiz_labels_from_aux(b["aux"])
        prompts, labels = [], []
        for g, mir in zip(b["frame"], b["mirrored"]):
            objs = [o for o in self.frame_objects[int(g)] if (o[1], o[2]) != HELD_OUT]
            vis = [o for o in objs if o[4]]
            colour, shape, bearing, visible = (vis[rng.integers(len(vis))] if vis and rng.random() < self.cfg.quiz_p_visible
                                               else objs[rng.integers(len(objs))])[1:]
            deg = -math.degrees(bearing) if mir else math.degrees(bearing)   # a mirrored picture swaps left and right
            labels.append(0 if not visible else (1 if deg > CENTRE_DEG else 3 if deg < -CENTRE_DEG else 2))
            prompts.append(self.qpara(colour, shape))
        return prompts, torch.tensor(labels)

    def quiz_examples(self, n: int = 8, seed: int = 1):
        """n training samples with quiz questions and answers, for looking at. Touches none of the random number streams
        training uses (data loader, quiz wording), so looking doesn't change the training result.
        -> (batch dict, prompts, labels)"""
        import copy
        idx = np.random.default_rng(seed).choice(len(self.ds), n, replace=False)
        saved = copy.deepcopy((self.ds.rng, self.ds.paraphraser, self.qpara.rng))
        b = _collate([self.ds[int(i)] for i in idx])
        prompts, labels = self.quiz_questions(b, np.random.default_rng(seed))
        self.ds.rng, self.ds.paraphraser, self.qpara.rng = saved
        return b, prompts, labels

    def quiz_logits(self, images, prompts, states, generator=None):
        """The quiz reads the driver at the FIRST refining step (pure noise in, t = 1): it has seen picture + words +
        speed, but no part of an answer's actions, so it can't cheat by reading the turn from them."""
        B = len(prompts)
        inp = self.prep_in(images, prompts, states)
        noise = torch.randn(B, 10, self.policy.config.max_action_dim, device=self.dev, generator=generator)
        return self.quiz(self.expert_out(inp, noise, torch.ones(B, device=self.dev)).mean(1))

    # ---------------------------------------------------------------------------------------------
    def train(self, steps: int | None = None):
        """Train up to cfg.steps (or `steps` more). Yields a log row every cfg.log_every steps."""
        c, m = self.cfg, self.m
        stop = min(c.steps, self.step + steps) if steps else c.steps
        m.train(); self.vw.get_vlm_model().vision_model.eval()
        t0 = time.time(); acc = {"flow": [], "quiz": [], "hit": []}
        while self.step < stop:
            for b in self.loader:
                for g, lr0 in zip(self.opt.param_groups, self.base_lrs):
                    g["lr"] = lr0 * self.lr_factor(self.step)
                B = len(b["task"])
                act = torch.zeros(B, 10, self.policy.config.max_action_dim, device=self.dev); act[:, :, :3] = b["action"].to(self.dev)
                with autocast(self.dev):
                    inp = self.prep_in(b["image"], b["task"], b["state"])
                    flow = m.forward(*inp, act)[:, :, :3].mean()   # flow-matching loss: "how wrong was the predicted correction?"
                    loss = flow
                    if self.quiz is not None:
                        prompts, lab = self.quiz_questions(b, self.qrng)
                        lg = self.quiz_logits(b["image"], prompts, b["state"])
                        lab = lab.to(self.dev)
                        ql = F.cross_entropy(lg.float(), lab)
                        loss = flow + c.quiz_weight * ql
                        acc["quiz"].append(ql.item()); acc["hit"].append((lg.argmax(1) == lab).float().mean().item())
                self.opt.zero_grad(set_to_none=True); loss.backward()
                gn = torch.nn.utils.clip_grad_norm_([p for g in self.groups for p in g["params"]], 10.0)
                self.opt.step(); self.step += 1; acc["flow"].append(flow.item())
                if self.step % c.log_every == 0 or self.step == stop:
                    row = {"step": self.step, "driving_loss": float(np.mean(acc["flow"])), "grad_norm": float(gn),
                           "lr_factor": self.lr_factor(self.step), "minutes": (time.time() - t0) / 60}
                    if self.quiz is not None:
                        row.update(quiz_loss=float(np.mean(acc["quiz"])), quiz_train_acc=float(np.mean(acc["hit"])))
                    if self.quiz is not None and (self.step % c.val_every == 0 or self.step == stop):
                        row["quiz_val_acc"] = self.val_quiz()
                    acc = {"flow": [], "quiz": [], "hit": []}
                    self.log.append(row)
                    yield row
                if self.step >= stop:
                    break

    @torch.no_grad()
    def val_quiz(self):
        """Quiz accuracy on the validation drives (pictures it never trained on)."""
        self.m.eval(); hit = n = 0; vrng = np.random.default_rng(123)
        vl = torch.utils.data.DataLoader(self.val, batch_size=64, shuffle=False, num_workers=2, collate_fn=_collate)
        g = torch.Generator(device=self.dev).manual_seed(0)
        for b in vl:
            with autocast(self.dev):
                prompts, lab = self.quiz_questions(b, vrng)
                lg = self.quiz_logits(b["image"], prompts, b["state"], generator=g)
            hit += (lg.argmax(1).cpu() == lab).sum().item(); n += len(prompts)
        self.m.train(); self.vw.get_vlm_model().vision_model.eval()
        return hit / n

    def save(self, path):
        """Trained weights (driver + unlocked reader layers) -> path; the quiz head -> <path>_quiz_head.pt."""
        import json
        from pathlib import Path
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({n: p.detach().cpu() for n, p in self.m.named_parameters() if p.requires_grad}, path)
        from .vla import save_meta
        save_meta(path, self.policy)   # which head + input format (load_bc_policy reads it)
        if self.quiz is not None:
            torch.save(self.quiz.state_dict(), path.with_name(path.stem + "_quiz_head.pt"))
        json.dump({"config": asdict(self.cfg), "prep": self.prep_path, "steps_done": self.step, "log": self.log},
                  open(path.with_name(path.stem + "_train_log.json"), "w"), indent=1)
        return path


def plot_training(log, fig=None):
    """Live graphs: driving loss, quiz loss, quiz accuracy (training and validation)."""
    import matplotlib.pyplot as plt
    fig = fig or plt.figure(figsize=(15, 3.8))
    fig.clf()
    ax = fig.subplots(1, 3)
    s = [r["step"] for r in log]
    ax[0].plot(s, [r["driving_loss"] for r in log]); ax[0].set_title("driving loss (lower = closer to the demos)")
    ax[0].set_xlabel("step"); ax[0].set_yscale("log")
    if log and "quiz_loss" in log[0]:
        ax[1].plot(s, [r["quiz_loss"] for r in log], color="tab:orange"); ax[1].set_title("quiz loss"); ax[1].set_xlabel("step")
        ax[2].plot(s, [r["quiz_train_acc"] for r in log], label="training pictures")
        v = [(r["step"], r["quiz_val_acc"]) for r in log if r.get("quiz_val_acc") is not None]
        if v:
            ax[2].plot(*zip(*v), "o-", label="validation pictures (never trained on)")
        ax[2].axhline(0.25, color="grey", ls=":", label="random guessing (4 answers)")
        ax[2].set_ylim(0, 1); ax[2].set_title("quiz: share answered right"); ax[2].set_xlabel("step"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    return fig
