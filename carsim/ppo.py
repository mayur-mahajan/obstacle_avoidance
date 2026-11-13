"""PPO fine-tuning of the imitation-trained smolVLA policy (used by notebook 05).

One ROUND = collect experience, then learn from it:
  1. Collect: 8 simulators x 256 decisions (one decision = a 10-action chunk, of which 5 are driven,
     = 0.5 s). For every decision we store what the robot saw, the jittered refining chain it used,
     how likely that chain was (logp), the critic's guess (value) and the reward it got.
  2. Advantages: for each decision, "how much better did things go than the critic expected?"
     computed with GAE (Schulman et al. 2016), which blends short- and long-term evidence.
  3. Learn: a few passes over the batch in mini-batches of 32. The PPO loss makes better-than-
     expected chains more likely and worse ones less likely, but CLIPS the change (ratio outside
     1 +/- 0.2 earns nothing more) so one batch cannot move the policy far. The critic learns to
     predict the actual returns. The imitation ("practise the demos") loss on demo samples (notebook 03) is
     added, with a weight that is lowered over time. If the policy changed too much (KL estimate
     above a limit) the remaining passes of this round are skipped.
First rounds = critic warm-up: only the critic learns (the driving stays as it is) until its
predictions are sensible, so the first policy updates are not driven by a random critic.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

from .data import ActionScaler, NavChunkDataset, load_training_data
from .device import empty_cache, pick_device
from .rl_env import ParallelNav, RewardConfig
from .rl_policy import FlowRLPolicy
from .vla import load_bc_policy


@dataclass
class PPOConfig:
    # experience collection
    n_envs: int = 8                      # each simulator uses ~280 MB of GPU memory for rendering; 8 ~= 16 in speed
    decisions_per_round: int = 256       # per simulator -> 2048 decisions per round (128 gave too few finished drives per lesson)
    # Practice-layout dials (the standard rooms use 0.5 / 0.5). The imitation model fails mostly when the goal starts
    # hidden (and then often gets stuck on obstacles); lookalike objects were not a big cause in ordinary rooms. So the
    # goal starts hidden in 3 of 4 practice drives, and lookalikes appear at the standard rate. The TEST set always
    # uses standard layouts, so test results stay comparable between runs.
    practice_layout: dict = field(default_factory=lambda: {"p_goal_visible": 0.25, "p_hard_distractor": 0.5})
    exec_steps: int = 5                  # actions driven per 10-action chunk
    denoise_steps: int = 4
    gamma: float = 0.995                 # discount per 0.1 s step (looks ~20 s ahead)
    gae_lambda: float = 0.95
    # learning
    epochs: int = 2
    minibatch: int = 32
    micro_batch: int = 16                # processed 16 at a time to keep GPU memory low (same result)
    clip: float = 0.2
    target_kl: float = 0.05              # stop this round's passes early if the policy moved more
    actor_lr: float = 1e-6                # measured: 1e-5 moved the policy ~60x too far in one step
    critic_lr: float = 3e-4
    sigma_lr: float = 1e-3
    sigma_init: float = 0.08
    sigma_min: float = 0.02
    sigma_max: float = 0.2
    max_grad_norm: float = 1.0
    critic_warmup_rounds: int = 5
    # "practise the demos" (imitation) loss
    bc_batch: int = 16
    bc_mirror_p: float = 0.0             # mirroring of the demo samples (0: as in imitation training, notebook 03)
    bc_coef_start: float = 1.0
    bc_coef_end: float = 0.1
    bc_decay_rounds: int = 100
    # testing / saving
    eval_every: int = 10
    eval_episodes_per_env: int = 4       # fixed test set: 8 x 4 = 32 drives, no jitter
    eval_seed: int = 999
    seed: int = 0
    full_precision: bool = True          # fp32 everywhere: bf16 gave fake ratio noise (KL 0.012 with no learning)
    reward: RewardConfig = field(default_factory=RewardConfig)


class PPOTrainer:
    def __init__(self, cfg: PPOConfig, init_checkpoint, run_dir, prep_path, device: str | None = None):
        device = device or pick_device()
        self.cfg, self.device, self.run_dir = cfg, device, Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        prep, table, frames, eps = load_training_data(prep_path)
        self.scaler = ActionScaler(np.array(prep["action_limits"], np.float32))
        policy = load_bc_policy(init_checkpoint, device)
        self.rl = FlowRLPolicy(policy, self.scaler, cfg.denoise_steps, cfg.exec_steps, cfg.sigma_init,
                               cfg.sigma_min, cfg.sigma_max, device, cfg.full_precision)
        self.rl.setup_trainable()
        self.opt = torch.optim.AdamW([
            {"params": self.rl.actor_parameters(), "lr": cfg.actor_lr},
            {"params": list(self.rl.value_head.parameters()), "lr": cfg.critic_lr},
            {"params": [self.rl.log_sigma], "lr": cfg.sigma_lr}], betas=(0.9, 0.95), weight_decay=0.0)
        # the demos for the imitation loss (same augmentation as imitation training)
        ds = NavChunkDataset(table, frames, eps, prep["split"]["train"], self.scaler, chunk=policy.config.chunk_size,
                             mirror_p=cfg.bc_mirror_p, paraphrase=True)
        collate = lambda items: {"image": torch.stack([s["image"] for s in items]), "state": torch.stack([s["state"] for s in items]),
                                 "action": torch.stack([s["action"] for s in items]), "task": [s["task"] for s in items]}
        self._bc_loader = torch.utils.data.DataLoader(ds, batch_size=cfg.bc_batch, shuffle=True, num_workers=2, collate_fn=collate,
                                                      worker_init_fn=ds.worker_init, persistent_workers=True, drop_last=True)
        self._bc_iter = iter(self._bc_loader)
        self.venv = ParallelNav(cfg.n_envs, seed=cfg.seed, reward_cfg=cfg.reward, gamma=cfg.gamma, layout=cfg.practice_layout)
        self.obs = self.venv.reset()
        self.round = 0
        self.log: list[dict] = []           # one entry per round
        self.episodes: list[dict] = []      # every finished training episode
        self.evals: list[dict] = []         # fixed test-set results
        json.dump({"config": asdict(cfg), "init_checkpoint": str(init_checkpoint)}, open(self.run_dir / "config.json", "w"),
                  indent=1, default=str)

    # ------------------------------------------------------------------------------------------------
    def _bc_batch(self):
        try:
            return next(self._bc_iter)
        except StopIteration:
            self._bc_iter = iter(self._bc_loader)
            return next(self._bc_iter)

    def bc_coef(self):
        c = self.cfg
        f = min(1.0, max(0, self.round - c.critic_warmup_rounds) / c.bc_decay_rounds)
        return c.bc_coef_start + f * (c.bc_coef_end - c.bc_coef_start)

    def collect(self):
        """Run every simulator for ``decisions_per_round`` decisions with the jittered policy."""
        c, T, N = self.cfg, self.cfg.decisions_per_round, self.cfg.n_envs
        buf = {k: [] for k in ("images", "states", "prompts", "chain", "logp", "value", "reward", "n", "done", "trunc", "boot")}
        imgs, states, prompts = self.obs
        for _ in range(T):
            a = self.rl.act(imgs, prompts, states, stochastic=True)
            out = self.venv.step(a["actions"])
            boot = np.zeros(N, np.float32)
            tr = out["truncated"]
            if tr.any():  # hit the 60 s limit: the future still had value, so ask the critic about the last view
                idx = np.flatnonzero(tr)
                # (out["prompts"] already belongs to the NEXT episode after the auto-reset -> use this episode's prompt)
                boot[idx] = self.rl.value(out["final_images"][idx], [prompts[i] for i in idx], out["final_states"][idx]).numpy()
            for k, v in [("images", imgs), ("states", states), ("prompts", prompts), ("chain", a["chain"]), ("logp", a["logp"]),
                         ("value", a["value"]), ("reward", out["reward"]), ("n", out["n"]), ("done", out["done"]),
                         ("trunc", tr), ("boot", boot)]:
                buf[k].append(v)
            self.episodes += [dict(e, round=self.round) for e in out["episodes"] if e is not None]
            imgs, states, prompts = out["images"], out["states"], out["prompts"]
        self.obs = (imgs, states, prompts)
        last_v = self.rl.value(imgs, prompts, states).numpy()
        return buf, last_v

    def advantages(self, buf, last_v):
        """GAE. A decision of n steps discounts the next one by gamma**n. At a success the future is 0;
        at the time limit the critic's estimate of the last view is used ("boot"); either way the
        advantage chain is cut because the next observation belongs to a new episode."""
        c = self.cfg
        r, v = np.stack(buf["reward"]), torch.stack(buf["value"]).numpy()
        n, done, boot = np.stack(buf["n"]), np.stack(buf["done"]), np.stack(buf["boot"])
        T = len(r)
        adv, last = np.zeros_like(r), np.zeros(r.shape[1], np.float32)
        for t in reversed(range(T)):
            next_v = v[t + 1] if t + 1 < T else last_v
            next_v = np.where(done[t], boot[t], next_v)
            g = c.gamma ** n[t]
            delta = r[t] + g * next_v - v[t]
            last = delta + g * c.gae_lambda * (~done[t]) * last
            adv[t] = last
        return adv, adv + v

    def learn(self, buf, adv, ret):
        c = self.cfg
        T, N = adv.shape
        flat = lambda x: [y for row in x for y in row]
        images = np.concatenate(buf["images"]); states = np.concatenate(buf["states"]); prompts = flat(buf["prompts"])
        chain = torch.cat(buf["chain"]); old_logp = torch.cat(buf["logp"])
        adv_t = torch.as_tensor(adv.reshape(-1)); ret_t = torch.as_tensor(ret.reshape(-1))
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        warm = self.round < c.critic_warmup_rounds
        bc_coef = self.bc_coef()
        st = {k: [] for k in ("policy_loss", "value_loss", "bc_loss", "kl", "clip_frac")}
        stopped = False
        planned, done_updates = c.epochs * math.ceil(T * N / c.minibatch), 0
        for _ep in range(c.epochs):
            perm = torch.randperm(T * N)
            for i in range(0, T * N, c.minibatch):
                idx_all = perm[i:i + c.minibatch].numpy()
                self.opt.zero_grad(set_to_none=True)
                kls, clips, pgs, vls, bcs = [], [], [], [], []
                # micro-batches: same learning as one mini-batch of 32 (gradients add up), lower memory peak
                for j in range(0, len(idx_all), c.micro_batch):
                    idx = idx_all[j:j + c.micro_batch]
                    w = len(idx) / len(idx_all)
                    R = ret_t[idx].to(self.device)
                    if warm:  # critic only: no need to re-score the action chains
                        value = self.rl.value_train(images[idx], [prompts[q] for q in idx], states[idx])
                        v_loss = 0.5 * ((value - R) ** 2).mean()
                        (w * v_loss).backward()
                        vls.append(v_loss.item())
                        continue
                    logp, value = self.rl.evaluate(images[idx], [prompts[q] for q in idx], states[idx], chain[idx])
                    v_loss = 0.5 * ((value - R) ** 2).mean()
                    log_ratio = logp - old_logp[idx].to(self.device)
                    ratio = log_ratio.exp()
                    A = adv_t[idx].to(self.device)
                    pg = -torch.min(ratio * A, ratio.clamp(1 - c.clip, 1 + c.clip) * A).mean()
                    (w * (pg + v_loss)).backward()
                    with torch.no_grad():
                        kls.append(((ratio - 1) - log_ratio).mean().item())   # low-variance KL estimate (Schulman)
                        clips.append(((ratio - 1).abs() > c.clip).float().mean().item())
                    pgs.append(pg.item()); vls.append(v_loss.item())
                if not warm:
                    b = self._bc_batch()  # separate backward = lower peak memory (gradients add up)
                    bc = self.rl.bc_loss(b["image"], b["task"], b["state"], b["action"])
                    (bc_coef * bc).backward()
                    bcs.append(bc.item())
                    st["policy_loss"].append(np.mean(pgs)); st["bc_loss"].append(np.mean(bcs))
                    st["kl"].append(np.mean(kls)); st["clip_frac"].append(np.mean(clips))
                st["value_loss"].append(np.mean(vls))
                params = [p for g in self.opt.param_groups for p in g["params"]]
                torch.nn.utils.clip_grad_norm_(params, c.max_grad_norm)
                if warm:  # critic only: drop any gradient that reached the actor
                    for p in self.rl.actor_parameters() + [self.rl.log_sigma]:
                        p.grad = None
                self.opt.step()
                done_updates += 1
                if not warm and np.mean(st["kl"][-4:]) > c.target_kl:
                    stopped = True
                    break
            if stopped:
                break
        v = torch.stack(buf["value"]).numpy().reshape(-1)
        out = {k: float(np.mean(x)) if x else None for k, x in st.items()}
        out.update(explained_variance=float(1 - np.var(ret.reshape(-1) - v) / (np.var(ret) + 1e-8)), early_stop=stopped,
                   study_done=done_updates / planned,  # share of the planned study steps done before the KL guard stopped
                   bc_coef=bc_coef if not warm else None, critic_warmup=warm)
        return out

    def train_round(self):
        t0 = time.time()
        buf, last_v = self.collect()
        t1 = time.time()
        adv, ret = self.advantages(buf, last_v)
        empty_cache(self.device)  # acting and learning use differently-shaped memory blocks
        stats = self.learn(buf, adv, ret)
        recent = [e for e in self.episodes if e["round"] == self.round]
        last100 = self.episodes[-100:]
        entry = {"round": self.round, "decisions": int(self.cfg.n_envs * self.cfg.decisions_per_round),
                 "episodes_finished": len(recent),
                 "success_last100": float(np.mean([e["success"] for e in last100])) if last100 else None,
                 "return_last100": float(np.mean([e["return"] for e in last100])) if last100 else None,
                 "crash_episodes_last100": float(np.mean([e["collision_steps"] > 0 for e in last100])) if last100 else None,
                 "reward_parts_last100": {k: float(np.mean([e["parts"][k] for e in last100])) for k in last100[0]["parts"]} if last100 else None,
                 "sigma": self.rl.sigma().detach().cpu().numpy().round(4).tolist(),
                 "collect_s": round(t1 - t0, 1), "learn_s": round(time.time() - t1, 1), **stats}
        self.after_round(entry)             # hook for subclasses (e.g. the curriculum's level-up)
        self.log.append(entry)
        self.round += 1
        if self.round % self.cfg.eval_every == 0:
            res = self.evaluate()
            self.extra_evals()               # hook for subclasses (e.g. per-level test sets)
            self.save("latest")
            std = [e for e in self.evals if e.get("tag", "standard") == "standard"]
            if res["return"] >= max(e["return"] for e in std):
                self.save("best", optimizer=False)
        return entry

    def after_round(self, entry: dict):
        pass

    def extra_evals(self):
        pass

    @torch.no_grad()
    def evaluate(self, episodes_per_env: int | None = None, seed: int | None = None, layout: dict | None = None,
                 tag: str = "standard"):
        """The fixed test set: the same layouts every time, no jitter, exactly k drives per simulator
        (counting only the first k of each simulator avoids favouring quick drives).
        layout/tag: another fixed test set (e.g. one curriculum level); default = the standard layouts."""
        k = episodes_per_env or self.cfg.eval_episodes_per_env
        self.venv.reseed(self.cfg.eval_seed if seed is None else seed, layout=layout or {})
        imgs, states, prompts = self.venv.reset()
        per_env = [[] for _ in range(self.cfg.n_envs)]
        while min(len(p) for p in per_env) < k:
            a = self.rl.act(imgs, prompts, states, stochastic=False)
            out = self.venv.step(a["actions"])
            for i, e in enumerate(out["episodes"]):
                if e is not None and len(per_env[i]) < k:
                    per_env[i].append(e)
            imgs, states, prompts = out["images"], out["states"], out["prompts"]
        eps = [e for p in per_env for e in p]
        res = {"round": self.round, "tag": tag, "n": len(eps),
               "success": float(np.mean([e["success"] for e in eps])),
               "success_visible_start": float(np.mean([e["success"] for e in eps if e["start_visible"]] or [np.nan])),
               "success_hidden_start": float(np.mean([e["success"] for e in eps if not e["start_visible"]] or [np.nan])),
               "crash_episodes": float(np.mean([e["collision_steps"] > 0 for e in eps])),
               "return": float(np.mean([e["return"] for e in eps])),
               "seconds": float(np.mean([e["seconds"] for e in eps]))}
        self.evals.append(res)
        # back to fresh random layouts for training
        self.venv.reseed(self.cfg.seed * 7919 + self.round + 1, layout=self.cfg.practice_layout)
        self.obs = self.venv.reset()
        return res

    def save(self, tag: str = "latest", optimizer: bool = True):
        """latest.pt = everything needed to continue (~1.2 GB, overwritten); best.pt = best test-set score
        so far, without the optimizer (~0.4 GB). Only two files, so long runs don't fill the disk."""
        trainable = {n: p.detach().cpu() for n, p in self.rl.m.named_parameters() if p.requires_grad}
        ck = {"policy": trainable, "value_head": self.rl.value_head.state_dict(), "log_sigma": self.rl.log_sigma.detach().cpu(),
              "round": self.round, "evals": self.evals}
        if optimizer:
            ck["optimizer"] = self.opt.state_dict()
        torch.save(ck, self.run_dir / f"{tag}.pt")
        json.dump({"log": self.log, "evals": self.evals, "episodes": self.episodes}, open(self.run_dir / "history.json", "w"))

    def load(self, path):
        """Continue from a saved round (policy, critic, jitter size, optimizer, history)."""
        ck = torch.load(path, map_location="cpu")
        named = dict(self.rl.m.named_parameters())
        for n, p in ck["policy"].items():
            named[n].data.copy_(p.to(self.device))
        self.rl.value_head.load_state_dict(ck["value_head"])
        self.rl.log_sigma.data.copy_(ck["log_sigma"].to(self.device))
        if "optimizer" in ck:
            self.opt.load_state_dict(ck["optimizer"])
        self.round = ck["round"]
        hist = Path(path).parent / "history.json"
        if hist.exists():
            h = json.load(open(hist))
            self.log, self.evals, self.episodes = h["log"], h["evals"], h["episodes"]

    def close(self):
        self.venv.close()


# ----------------------------------------------------------------------------------------------------
def plot_progress(trainer, fig=None):
    """Live training graphs (redrawn every round in the notebook)."""
    import matplotlib.pyplot as plt

    log = trainer.log
    ev = [e for e in trainer.evals if e.get("tag", "standard") == "standard"]
    fig = fig or plt.figure(figsize=(15, 8))
    fig.clf()
    ax = fig.subplots(2, 3).ravel()
    r = [e["round"] for e in log]

    def line(a, key, label, **kw):
        ys = [e[key] if e.get(key) is not None else np.nan for e in log]
        a.plot(r, ys, label=label, **kw)

    line(ax[0], "success_last100", "practice drives (last 100, with jitter)")
    if ev:
        ax[0].plot([e["round"] for e in ev], [e["success"] for e in ev], "o-", label="fixed test set (no jitter)")
        ax[0].plot([e["round"] for e in ev], [e["success_hidden_start"] for e in ev], "s--", label="test: goal hidden at start")
    ax[0].set_title("Success rate"); ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
    line(ax[1], "return_last100", "practice")
    if ev:
        ax[1].plot([e["round"] for e in ev], [e["return"] for e in ev], "o-", label="test set")
    ax[1].set_title("Score per drive"); ax[1].legend(fontsize=8)
    line(ax[2], "crash_episodes_last100", "practice")
    if ev:
        ax[2].plot([e["round"] for e in ev], [e["crash_episodes"] for e in ev], "o-", label="test set")
    ax[2].set_title("Drives with a crash"); ax[2].set_ylim(0, 1); ax[2].legend(fontsize=8)
    line(ax[3], "explained_variance", "critic")
    ax[3].set_title("Critic accuracy (explained variance, 1 = perfect)"); ax[3].set_ylim(-1, 1)
    line(ax[4], "kl", "KL (how far the policy moved)"); line(ax[4], "clip_frac", "fraction clipped")
    line(ax[4], "study_done", "share of study done before KL stop", ls=":")
    ax[4].set_title("Size of each policy update"); ax[4].legend(fontsize=8)
    ax[5].plot(r, [e["sigma"] for e in log], label=["vx", "vy", "wz"])
    ax[5].set_title("Jitter size sigma (exploration)"); ax[5].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("round"); a.grid(alpha=0.3)
    fig.tight_layout()
    return fig
