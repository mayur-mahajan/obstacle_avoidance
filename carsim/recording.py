"""Turning driving episodes into a LeRobot dataset.

One *frame* of the dataset = one control step (0.1 s):
  * what the car SAW just before acting  -> ``observation.images.front`` (256x256 RGB) + ``observation.state``
  * what the driver DID at that moment     -> ``action`` (vx, vy, wz)
  * the instruction for the whole episode  -> ``task`` (the prompt text)
  * extra "privileged" simulator facts     -> ``sim.*`` columns (never shown to the policy as input;
    used for analysis and, later, for an auxiliary training loss)

The pairing matters: the action stored with an image must be the action taken *after* seeing that
image. The recorder therefore stores (obs_before_step, action) - never the image after the step.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FPS = 10  # control rate = dataset frame rate
IMAGE_KEY = "observation.images.front"

FEATURES = {
    IMAGE_KEY: {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channels"]},
    # Measured body velocity. The policy may use it to know "am I already moving?".
    "observation.state": {"dtype": "float32", "shape": (3,), "names": ["vx", "vy", "wz"]},
    # The (ramped) command that was sent to the car.
    "action": {"dtype": "float32", "shape": (3,), "names": ["vx", "vy", "wz"]},
    # ---- privileged simulator ground truth (NOT a policy input) ----
    "sim.pose": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "yaw"]},
    # Goal relative to the car: bearing (rad, + = goal is to the left), centre distance (m), bumper-to-surface distance (m)
    "sim.goal_rel": {"dtype": "float32", "shape": (3,), "names": ["bearing", "distance", "surface_distance"]},
    "sim.geodesic_dist": {"dtype": "float32", "shape": (1,), "names": ["geodesic_dist"]},
    "sim.goal_visible": {"dtype": "float32", "shape": (1,), "names": ["goal_visible"]},
    "sim.collision": {"dtype": "float32", "shape": (1,), "names": ["collision"]},
    "sim.episode_seed": {"dtype": "int64", "shape": (1,), "names": ["seed"]},
}


def make_frame(obs: dict, action, info: dict, spec, prompt: str) -> dict:
    """Build one dataset frame from the observation *before* the step, the action taken, and the env
    ``info`` that belongs to that same observation."""
    x, y, yaw = info["pose"]
    _shape, _colour, gx, gy = spec.goal
    bearing = (math.atan2(gy - y, gx - x) - yaw + math.pi) % (2 * math.pi) - math.pi
    geo = info["geodesic_dist"]
    return {
        IMAGE_KEY: obs["image"],
        "observation.state": np.asarray(obs["state"], dtype=np.float32),
        "action": np.asarray(action, dtype=np.float32),
        "sim.pose": np.array([x, y, yaw], dtype=np.float32),
        "sim.goal_rel": np.array([bearing, math.hypot(gx - x, gy - y), info["goal_surface_dist"]], dtype=np.float32),
        # geodesic distance is +inf when the car centre is inside an inflated obstacle; cap it for clean stats
        "sim.geodesic_dist": np.array([min(float(geo), 20.0)], dtype=np.float32),
        "sim.goal_visible": np.array([float(info["goal_visible"])], dtype=np.float32),
        "sim.collision": np.array([float(info["collision"])], dtype=np.float32),
        "sim.episode_seed": np.array([spec.seed], dtype=np.int64),
        "task": prompt,
    }


@dataclass
class EpisodeRecord:
    """Frames of the episode being driven, kept in RAM until the driver decides keep / discard.
    (600 frames x 196 KB = 118 MB at most, so memory is no problem, and nothing half-finished ever
    touches the dataset on disk.)"""

    seed: int
    suggested_prompt: str
    prompt: str
    frames: list = field(default_factory=list)


def open_dataset(root: Path, repo_id: str):
    """Create the dataset, or reopen it for appending if it already exists."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    kw = dict(image_writer_threads=4)
    if (root / "meta" / "info.json").exists():
        return LeRobotDataset.resume(repo_id, root=root, **kw)
    root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(repo_id, fps=FPS, features=FEATURES, root=root, robot_type="mecanum_car",
                                 use_videos=True, metadata_buffer_size=1, **kw)


def save_episode(ds, record: EpisodeRecord) -> int:
    """Write the buffered frames as one episode. Returns the new episode's index."""
    index = ds.meta.total_episodes
    for fr in record.frames:
        ds.add_frame(fr)
    ds.save_episode()
    return index


def append_log(root: Path, row: dict) -> None:
    """Every attempt (kept or not) goes into ``attempts.jsonl`` next to the dataset, so we can later
    study *all* driving, including the discarded episodes."""
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    with open(Path(root).parent / f"{Path(root).name}_attempts.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def read_log(root: Path) -> list[dict]:
    p = Path(root).parent / f"{Path(root).name}_attempts.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


# -------------------------------------------------------------------------------------------------
# Reading back (for analysis notebooks)
# -------------------------------------------------------------------------------------------------
def load_table(root: Path):
    """All numeric columns of a dataset as one pandas DataFrame (no video decoding, so it is fast).
    Vector columns are split into one column per component, e.g. ``action.vx``, ``sim.goal_rel.bearing``."""
    import pandas as pd

    root = Path(root)
    df = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").glob("*/*.parquet"))], ignore_index=True)
    info = json.loads((root / "meta" / "info.json").read_text())
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    task_of = {int(i): t for t, i in zip(tasks.index, tasks["task_index"])}
    out = df[["index", "episode_index", "frame_index", "timestamp"]].copy()
    out["task"] = df["task_index"].map(task_of)
    for key, feat in info["features"].items():
        if key not in df.columns or feat["dtype"] in ("video", "image") or key in out.columns:
            continue
        vals = np.stack(df[key].to_numpy()) if feat["shape"] != [1] or df[key].dtype == object else df[key].to_numpy()
        vals = vals.reshape(len(df), -1)
        names = feat.get("names") or [str(i) for i in range(vals.shape[1])]
        if vals.shape[1] == 1:
            out[key] = vals[:, 0]
        else:
            for i, n in enumerate(names):
                out[f"{key}.{n}"] = vals[:, i]
    return out


def episode_images(ds, episode_index: int, every: int = 1) -> np.ndarray:
    """Decode the camera frames of one episode -> uint8 array (T, H, W, 3)."""
    ep = ds.meta.episodes[episode_index]
    idx = range(int(ep["dataset_from_index"]), int(ep["dataset_to_index"]), every)
    return np.stack([(ds[i][IMAGE_KEY].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for i in idx])
