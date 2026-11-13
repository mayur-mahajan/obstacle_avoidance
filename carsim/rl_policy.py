"""smolVLA made ready for PPO: a flow-matching action head whose action has a computable probability.

How the head normally makes an action chunk (10 actions x 32 numbers, of which we use the first 3):
start from random noise x and refine it in K small steps, t = 1 -> 0:

    x  <-  x + dt * v(x, t)          (dt = -1/K;  v = the action expert's "which way to move" output)

Every step is deterministic, so nobody can say how LIKELY the final action was - but PPO needs exactly
that ("make good actions more likely"). The fix used by ReinFlow (Zhang et al. 2025) and pi-RL's
Flow-Noise: add a small, known random jitter at every step,

    x  <-  x + dt * v(x, t) + sigma * noise

Each jittered step is a bell curve (Gaussian) around the deterministic step, whose probability has a
simple formula; the probability of the whole chain is the product of the steps. The jitter is also how
the robot explores. sigma is learned (bounded), one value per action component (vx, vy, wz).

Only the numbers that are actually driven get jitter: the first ``exec_steps`` actions of the chunk,
components (vx, vy, wz). The rest of the chunk is refined deterministically, as before.

The value head ("critic") predicts the expected future score from the locked reader's features of
(image + prompt + speed). The VLM itself stays frozen; the trainable parts are the action expert,
the value head and sigma.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .vla import AttentionPoolProbe, format_prompts, prepare_inputs

LOG_2PI = math.log(2 * math.pi)


class FlowRLPolicy(nn.Module):
    def __init__(self, policy, scaler, denoise_steps: int = 4, exec_steps: int = 5, sigma_init: float = 0.08,
                 sigma_min: float = 0.02, sigma_max: float = 0.2, device: str | None = None, full_precision: bool = False):
        super().__init__()
        device = device or next(policy.parameters()).device
        self.policy, self.m, self.scaler, self.device = policy, policy.model, scaler, device
        self.K, self.exec_steps, self.chunk = denoise_steps, exec_steps, policy.config.chunk_size
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.log_sigma = nn.Parameter(torch.full((3,), math.log(sigma_init), device=device))
        self.value_head = AttentionPoolProbe(self.m.vlm_with_expert.config.text_config.hidden_size, 1,
                                             dropout=0.0).to(device)
        mask = torch.zeros(self.chunk, policy.config.max_action_dim, dtype=torch.bool, device=device)
        mask[:exec_steps, :3] = True
        self.noise_mask = mask  # which numbers of the chunk are jittered (= the ones we drive with)
        self.m.eval()           # no dropout anywhere; gradients still flow to trainable parameters
        # full_precision: run everything in fp32. bf16 arithmetic gives slightly different numbers for the
        # same sample in batches of different size, which shows up as fake policy change in PPO's ratios.
        self.amp = not full_precision
        if full_precision:
            self.m.float()

    def _autocast(self):
        """bf16 autocast only on NVIDIA GPUs and only if full_precision is off (see carsim/device.py)."""
        import contextlib
        from .device import autocast
        return autocast(self.device) if self.amp else contextlib.nullcontext()

    # ---- parameters -------------------------------------------------------------------------------
    def actor_parameters(self):
        m = self.m
        mods = [m.vlm_with_expert.lm_expert, m.state_proj, m.action_in_proj, m.action_out_proj,
                m.action_time_mlp_in, m.action_time_mlp_out]
        return [p for mod in mods for p in mod.parameters()]

    def setup_trainable(self):
        """Freeze everything except the action expert (fp32 master copies), the value head and sigma."""
        for p in self.m.parameters():
            p.requires_grad = False
        for p in self.actor_parameters():
            p.data = p.data.float()
            p.requires_grad = True

    def sigma(self):
        return self.log_sigma.exp().clamp(self.sigma_min, self.sigma_max)

    # ---- inputs -----------------------------------------------------------------------------------
    def inputs(self, images, prompts, states):
        """images (B,H,W,3) uint8, prompts [B], states (B,3) raw speeds -> smolVLA inputs."""
        img = torch.as_tensor(np.asarray(images)).to(self.device).permute(0, 3, 1, 2).float() / 255
        st = torch.as_tensor(self.scaler.scale(np.asarray(states, np.float32)))
        return prepare_inputs(self.policy, img, format_prompts(self.policy, prompts), st, device=self.device, fixed_length=True)

    @torch.no_grad()
    def _prefix(self, inp):
        """Run the frozen VLM once: its key/value cache (for the action expert) and its output features
        (for the value head). Nothing here is trained, so no gradients are kept."""
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        m = self.m
        images, img_masks, lang_tokens, lang_masks, state = inp
        with self._autocast():
            embs, pad, att = m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
            att2d = make_att_2d_masks(pad, att)
            pos = torch.cumsum(pad, dim=1) - 1
            (out, _), cache = m.vlm_with_expert.forward(attention_mask=att2d, position_ids=pos, past_key_values=None,
                                                        inputs_embeds=[embs, None], use_cache=True)
        return pad, cache, out.float()

    def _velocity(self, x, t, pad, cache):
        tt = torch.full((x.shape[0],), t, device=self.device, dtype=torch.float32)
        with self._autocast():
            return self.m.denoise_step(prefix_pad_masks=pad, past_key_values=cache, x_t=x, timestep=tt).float()

    def _step_logp(self, x_next, mean, sigma_full):
        z = ((x_next - mean) / sigma_full)[:, self.noise_mask]
        return (-0.5 * z ** 2 - torch.log(sigma_full[self.noise_mask]) - 0.5 * LOG_2PI).sum(-1)

    def _sigma_full(self):
        s = torch.ones(self.chunk, self.policy.config.max_action_dim, device=self.device)
        s[:, :3] = self.sigma()
        return s

    # ---- acting -----------------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, images, prompts, states, stochastic: bool = True, generator=None):
        """-> dict: actions (B, exec_steps, 3) RAW (ready for the env), chain (B, K+1, chunk, 32),
        logp (B,), value (B,)."""
        inp = self.inputs(images, prompts, states)
        pad, cache, feats = self._prefix(inp)
        B = len(prompts)
        x = torch.randn(B, self.chunk, self.policy.config.max_action_dim, device=self.device, generator=generator)
        chain, logp, sig = [x], torch.zeros(B, device=self.device), self._sigma_full()
        for k in range(self.K):
            mean = x + (-1.0 / self.K) * self._velocity(x, 1.0 - k / self.K, pad, cache)
            if stochastic:
                x = mean + self.noise_mask * sig * torch.randn(mean.shape, device=self.device, generator=generator)
                logp += self._step_logp(x, mean, sig)
            else:
                x = mean
            chain.append(x)
        value = self.value_head(feats, pad)[0][:, 0]
        a = x[:, : self.exec_steps, :3].clamp(-1, 1)
        return {"actions": self.scaler.unscale(a.cpu().numpy()), "chain": torch.stack(chain, 1).cpu(),
                "logp": logp.cpu(), "value": value.cpu()}

    @torch.no_grad()
    def value(self, images, prompts, states):
        pad, _cache, feats = self._prefix(self.inputs(images, prompts, states))
        return self.value_head(feats, pad)[0][:, 0].cpu()

    # ---- learning ---------------------------------------------------------------------------------
    def value_train(self, images, prompts, states):
        """The critic's guess WITH gradients (for the critic warm-up, where the actions aren't re-scored)."""
        pad, _cache, feats = self._prefix(self.inputs(images, prompts, states))
        return self.value_head(feats, pad)[0][:, 0]

    def evaluate(self, images, prompts, states, chain):
        """Re-score a stored chain with the CURRENT parameters (with gradients): log-probability of
        every jittered step, and the value. chain: (B, K+1, chunk, 32)."""
        inp = self.inputs(images, prompts, states)
        pad, cache, feats = self._prefix(inp)
        chain = chain.to(self.device)
        sig = self._sigma_full()
        logp = torch.zeros(chain.shape[0], device=self.device)
        for k in range(self.K):
            x = chain[:, k]
            mean = x + (-1.0 / self.K) * self._velocity(x, 1.0 - k / self.K, pad, cache)
            logp = logp + self._step_logp(chain[:, k + 1], mean, sig)
        value = self.value_head(feats, pad)[0][:, 0]
        return logp, value

    def bc_loss(self, images, prompts, states, actions):
        """The usual imitation (flow-matching) loss on demo samples - keeps RL close to the demonstrated
        driving. images (B,3,H,W) float [0,1], states (B,3) SCALED, actions (B, chunk, 3) SCALED."""
        inp = prepare_inputs(self.policy, images, format_prompts(self.policy, prompts), states, device=self.device)
        act = torch.zeros(len(prompts), self.chunk, self.policy.config.max_action_dim, device=self.device)
        act[:, :, :3] = actions.to(self.device)
        with self._autocast():
            return self.m.forward(*inp, act)[:, :, :3].mean()
