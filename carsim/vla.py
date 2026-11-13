"""Helpers around smolVLA: loading it, preparing our inputs, reading its internal features, probing.

smolVLA = a vision-language model (SmolVLM2, cut to its first 16 layers) + an "action expert".
For one observation the VLM sees a sequence of "prefix" tokens:

    [ 64 image tokens | prompt tokens (~10-20) | 1 state token ]

Image and prompt tokens attend to each other in both directions, so after a few layers the image
tokens can "know" which object the prompt is about. The action expert reads the VLM's keys/values
at every layer and turns noise into an action chunk (flow matching).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

SMOLVLA_ID = "lerobot/smolvla_base"
IMAGE_TOKENS = 64  # SigLIP patches of the 512x512 image, merged 4x4 by the connector
MAX_DIM = 32  # smolVLA pads state and action vectors to 32 numbers


def load_smolvla(device: str | None = None):
    """The pretrained policy, in eval mode, on ``device`` (default: carsim.device.pick_device()). The first call
    downloads ~0.9 GB. Its weights are stored in bf16; on NVIDIA GPUs bf16 autocast handles the mix, elsewhere
    (Apple Silicon, CPU) the whole model is converted to fp32 so that every operation sees one number format."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from .device import device_type, pick_device
    device = device or pick_device()
    policy = SmolVLAPolicy.from_pretrained(SMOLVLA_ID)
    if device_type(device) != "cuda":
        policy = policy.float()
    return policy.to(device).eval()


def prepare_inputs(policy, images, tasks, states, device=None, fixed_length: bool = False):
    """Our sample format -> smolVLA's model inputs.

    images: (B, 3, H, W) float in [0, 1];  tasks: list of B strings;  states: (B, 3) scaled speeds.
    Returns (images_list, image_masks, lang_tokens, lang_masks, state32), the arguments of
    ``VLAFlowMatching.embed_prefix``/``forward``/``sample_actions``.
    fixed_length: pad every prompt to the tokenizer's max length instead of the longest in the batch, so
    a sample gives the same numbers whatever batch it is in (needed for RL's probability ratios)."""
    from lerobot.policies.common.vla_utils import resize_with_pad

    cfg = policy.config
    device = device or next(policy.parameters()).device
    img = resize_with_pad(images.to(device), cfg.resize_imgs_with_padding[1], cfg.resize_imgs_with_padding[0], pad_value=0)
    img = img * 2.0 - 1.0  # SigLIP expects [-1, 1]
    tok = policy.model.vlm_with_expert.processor.tokenizer(
        [t if t.endswith("\n") else t + "\n" for t in tasks], padding="max_length" if fixed_length else "longest", return_tensors="pt",
        max_length=cfg.tokenizer_max_length, truncation=True)
    B = img.shape[0]
    state = torch.zeros(B, cfg.max_state_dim, device=device)   # 32 (smolVLA) or 3 (car head)
    state[:, : states.shape[1]] = states.to(device)
    return ([img], [torch.ones(B, dtype=torch.bool, device=device)], tok.input_ids.to(device),
            tok.attention_mask.bool().to(device), state)


@torch.no_grad()
def prefix_features(policy, inputs, which: str = "final", n_layers: int | None = None):
    """Frozen-VLM features of the prefix tokens.

    which="vision": the 64 image tokens straight out of the vision encoder + connector. They have
                    NOT seen the prompt yet (the vision encoder only looks at the image).
    which="final":  the image + prompt tokens after the VLM layers (all 16, or the first ``n_layers``),
                    plus the final norm; here the image tokens have been mixed with the prompt.
    Returns (features (B, L, 960) float32, mask (B, L) bool = real token, not padding)."""
    m = policy.model
    images, img_masks, lang_tokens, lang_masks, state = inputs
    if which == "vision":
        f = m.vlm_with_expert.embed_image(images[0])
        return f.float(), torch.ones(f.shape[:2], dtype=torch.bool, device=f.device)
    vw = m.vlm_with_expert
    keep = vw.num_vlm_layers
    vw.num_vlm_layers = n_layers or keep  # run only the first n layers (restored below)
    try:
        out, pad = _prefix_forward(m, images, img_masks, lang_tokens, lang_masks, state)
    finally:
        vw.num_vlm_layers = keep
    out, pad = out[:, :-1], pad[:, :-1]  # drop the state token: the probe must not peek at the speed
    return out.float(), pad


def _prefix_forward(m, images, img_masks, lang_tokens, lang_masks, state):
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    from .device import autocast
    with autocast(images[0].device):
        embs, pad, att = m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        att2d = make_att_2d_masks(pad, att)
        pos = torch.cumsum(pad, dim=1) - 1
        # use_cache=True makes every layer run plain self-attention over the prefix (with use_cache=False the
        # cross-attention layers would expect action-expert inputs too). We ignore the returned cache.
        (out, _), _ = m.vlm_with_expert.forward(attention_mask=att2d, position_ids=pos, past_key_values=None,
                                                inputs_embeds=[embs, None], use_cache=True)
    return out, pad


class AttentionPoolProbe(nn.Module):
    """The smallest sensible "reader" of a set of tokens: one learned query attends over all tokens
    (so it can pick out the goal wherever it is in the image), then a linear layer maps the result to
    the targets. With only ~0.5 M parameters it cannot compute much by itself, so a good score means
    the information is already present in the features."""

    def __init__(self, dim: int, n_out: int, d: int = 256, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, d))
        self.query = nn.Parameter(torch.randn(1, 1, d) / math.sqrt(d))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Linear(d, n_out)

    def forward(self, feats, mask):
        x = self.inp(feats)
        pooled, w = self.attn(self.query.expand(len(x), -1, -1), x, x, key_padding_mask=~mask)
        return self.out(pooled[:, 0]), w[:, 0]


# -------------------------------------------------------------------------------------------------
# Ground truth about every object in view (not just the goal), for the language test
# -------------------------------------------------------------------------------------------------
def objects_in_view(spec, pose, env_cfg=None, grid=None):
    """For each goal-type object of the episode: (index, colour, shape, bearing, visible)."""
    from .world import EnvConfig, Grid, goal_in_view, occupancy

    env_cfg = env_cfg or EnvConfig()
    grid = grid or Grid(env_cfg)
    x, y, yaw = pose
    out = []
    for k, (shape, colour, ox, oy) in enumerate(spec.objects):
        occ = occupancy(grid, spec.objects, spec.obstacles, skip_object=k)
        bearing = (math.atan2(oy - y, ox - x) - yaw + math.pi) % (2 * math.pi) - math.pi
        out.append((k, colour, shape, bearing, bool(goal_in_view(env_cfg, grid, occ, pose, (ox, oy)))))
    return out


# -------------------------------------------------------------------------------------------------
# The reader input format: present our inputs the way SmolVLM2 was trained to see them
# -------------------------------------------------------------------------------------------------
def set_input_format(policy, markers: bool = True, causal: bool = True, unscaled: bool = True, question: bool = False):
    """How the picture and words are shown to the reader (SmolVLM2). Three separate switches:
      markers   image marker tokens around the 64 image tokens (SmolVLM2 was trained with them; smolVLA leaves them out)
      causal    one-way attention inside the prefix: each token sees only the tokens before it (how SmolVLM2 was
                trained); False = smolVLA's original two-way attention between picture and words
      question  (not used any more) an appended "Where is the <colour> <shape>?"; must stay False
      unscaled  undo smolVLA's x sqrt(960) = x31 on the picture and prompt-word embeddings (lerobot modeling_smolvla.py
                embed_prefix). SmolVLM2 (a Llama-type model) never scales its inputs; pi0's Gemma does, which is probably
                where the step came from. The image marker tokens are not scaled by smolVLA, so they are left alone.
    Reader input fix (notebook 04, section 3): markers + causal + unscaled together -> the reader
    tells WHICH of two objects the words name 91% of the time (smolVLA as published: 50%; any one of the three missing:
    51-54%). The question is not needed.

    All off = smolVLA exactly as published. Call once, before training/loading."""
    assert not question, "the appended question is not part of the recipe any more"
    m = policy.model
    m.add_image_special_tokens = markers
    policy.config.add_image_special_tokens = markers
    if causal and not getattr(m, "_format_fixed", False):
        orig = m.embed_prefix

        def causal_embed_prefix(*a, **k):
            embs, pad, att = orig(*a, **k)
            return embs, pad, torch.ones_like(att)

        m.embed_prefix = causal_embed_prefix
        m._format_fixed = True
    assert causal or not getattr(m, "_format_fixed", False), "causal prefix already applied to this model"
    if unscaled and not getattr(m, "_unscaled", False):
        vw = m.vlm_with_expert
        k = math.sqrt(vw.config.text_config.hidden_size)
        embed_image, embed_words = vw.embed_image, vw.embed_language_tokens
        vw.embed_image = lambda img: embed_image(img) / k                     # smolVLA then multiplies by k: net x1
        # prompt words arrive as a (batch, length) tensor and are multiplied by k; the marker tokens arrive as a
        # 1-D tensor and are NOT multiplied by smolVLA, so they must not be divided either
        vw.embed_language_tokens = lambda t: embed_words(t) / k if t.dim() == 2 else embed_words(t)
        m._unscaled = True
    assert unscaled or not getattr(m, "_unscaled", False), "unscaled inputs already applied to this model"
    policy._input_format = {"markers": markers, "causal": causal, "question": question, "unscaled": unscaled}
    return policy


def apply_reader_fix(policy):
    """SmolVLM2's own input format (the recipe): image markers + one-way attention + no x31 scaling."""
    return set_input_format(policy, markers=True, causal=True, question=False, unscaled=True)


def format_prompts(policy, tasks):
    """The prompts as the model expects them (unchanged in the current recipe; kept as the one place to change it)."""
    return list(tasks)


# -------------------------------------------------------------------------------------------------
# The car action expert (a "custom head" with smolVLA's design)
# -------------------------------------------------------------------------------------------------
CAR_DIM = 3   # vx, vy, wz (actions) and the 3 measured speeds (state)


def make_car_head(policy, seed: int = 0):
    """Replace smolVLA's arm-pretrained action expert by a FRESH one with the same architecture, built for the car:

      same:      16 transformer layers paired with the reader's 16 layers, width 720, alternating joint attention and
                 cross-attention to the reader, flow-matching time embedding - exactly smolVLA's expert design
      different: 1) random starting weights (no robot-arm pretraining)
                 2) actions are 3 numbers (vx, vy, wz) instead of 32 padded slots; the robot state is 3 numbers too
    """
    m = policy.model
    vw = m.vlm_with_expert
    torch.manual_seed(seed)
    vw.lm_expert.apply(vw.lm_expert._init_weights)            # the transformer: Hugging Face's standard initialisation
    dev, hid, vhid = next(m.parameters()).device, vw.expert_hidden_size, vw.config.text_config.hidden_size
    m.action_in_proj = nn.Linear(CAR_DIM, hid).to(dev)
    m.action_out_proj = nn.Linear(hid, CAR_DIM).to(dev)
    m.state_proj = nn.Linear(CAR_DIM, vhid).to(dev)
    m.action_time_mlp_in.reset_parameters(); m.action_time_mlp_out.reset_parameters()
    policy.config.max_action_dim = CAR_DIM
    policy.config.max_state_dim = CAR_DIM
    policy._head = "car"
    return policy


def action_dim(policy) -> int:
    """Width of the action vectors the expert works with: 32 (smolVLA, padded) or 3 (car head)."""
    return policy.config.max_action_dim


def meta_path(checkpoint):
    from pathlib import Path
    c = Path(checkpoint)
    return c.with_name(c.stem + "_meta.json")


def save_meta(checkpoint, policy):
    """Sidecar file: which head and input format a checkpoint needs (read by load_bc_policy)."""
    import json
    json.dump({"head": getattr(policy, "_head", "smolvla"), "input_format": getattr(policy, "_input_format", None)},
              open(meta_path(checkpoint), "w"), indent=1)


def load_bc_policy(checkpoint, device: str | None = None, chunk: int = 10):
    """smolVLA + our imitation-trained weights (a dict of the trained parameters, saved in fp32).
    The _meta.json file next to the checkpoint (written by BCTrainer.save) says which head and input format it needs."""
    import json
    from .device import pick_device
    device = device or pick_device()
    if not meta_path(checkpoint).exists():
        raise FileNotFoundError(f"{meta_path(checkpoint)} is missing: train the checkpoint with notebook 04 (BCTrainer.save writes it)")
    meta = json.load(open(meta_path(checkpoint)))
    policy = load_smolvla(device)
    policy.config.chunk_size = chunk
    if meta["head"] == "car":
        make_car_head(policy)
    set_input_format(policy, **meta["input_format"])
    sd = torch.load(checkpoint, map_location="cpu")
    named = dict(policy.model.named_parameters())
    missing = [n for n in sd if n not in named]
    if missing:
        raise KeyError(f"checkpoint has unknown parameters, e.g. {missing[:3]}")
    bad = [n for n, p in sd.items() if named[n].shape != p.shape]
    if bad:
        raise ValueError(f"checkpoint shapes do not match the model, e.g. {bad[:3]}")
    for n, p in sd.items():
        named[n].data = p.to(device).float()
    return policy
