"""Which hardware to use, in one place, so the same code runs on an NVIDIA GPU, an Apple Silicon Mac or a plain CPU.

  pick_device()   "cuda" (NVIDIA GPU) > "mps" (Apple Silicon GPU) > "cpu". Override with CARSIM_DEVICE=cpu|cuda|mps.
  autocast(dev)   the fast-maths context for the neural network:
                    cuda -> bf16 autocast (half the memory, several times faster on tensor cores)
                    mps / cpu -> full precision (bf16 support on Apple GPUs is incomplete in PyTorch; full precision
                    is slower but always correct)
  sim_backend()   where Genesis runs its physics: the GPU if there is one (CUDA, or Metal on a Mac), else the CPU.
  describe()      one-line summary for the notebooks.

Everything else in the project asks these functions instead of writing "cuda" by hand.
"""
from __future__ import annotations

import contextlib
import os


def pick_device() -> str:
    import torch

    forced = os.environ.get("CARSIM_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_type(device) -> str:
    """'cuda', 'mps' or 'cpu' for a device string or torch.device (e.g. 'cuda:0' -> 'cuda')."""
    return str(device).split(":")[0]


def autocast(device):
    """bf16 mixed precision on NVIDIA GPUs; a do-nothing context elsewhere (full precision)."""
    import torch

    if device_type(device) == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def synchronize(device) -> None:
    """Wait for queued GPU work to finish (needed for honest timing)."""
    import torch

    t = device_type(device)
    if t == "cuda":
        torch.cuda.synchronize()
    elif t == "mps":
        torch.mps.synchronize()


def empty_cache(device) -> None:
    import torch

    t = device_type(device)
    if t == "cuda":
        torch.cuda.empty_cache()
    elif t == "mps":
        torch.mps.empty_cache()


def sim_backend(prefer: str = "gpu"):
    """Genesis backend: gs.gpu (CUDA on Linux/Windows, Metal on Apple Silicon) when a GPU exists, else gs.cpu.
    prefer="cpu" always gives gs.cpu (the RL arena uses CPU physics: faster for this small scene, see carsim/rl_env.py)."""
    import genesis as gs

    if prefer == "cpu" or pick_device() == "cpu":
        return gs.cpu
    return gs.gpu


def describe() -> str:
    import platform

    import torch

    d = pick_device()
    if d == "cuda":
        p = torch.cuda.get_device_properties(0)
        what = f"{p.name}, {p.total_memory / 1024**3:.1f} GB"
    elif d == "mps":
        what = f"Apple Silicon GPU (Metal), {platform.machine()}"
    else:
        what = f"CPU only ({platform.processor() or platform.machine()}): training will be very slow"
    return f"device = {d}: {what} | PyTorch {torch.__version__} | {platform.system()} {platform.release()}"
