"""Turning a recorded LeRobot dataset into training samples.

Pipeline (see notebook 03 for the why of every step):

    LeRobot dataset ──decode once──► frame cache (uint8 .npy in RAM)
                                         │
    split by EPISODE (train / val) ──────┤
                                         ▼
    NavChunkDataset[i] = {image, state, action chunk (+pad mask), prompt, aux targets}
        with, for training only:  mirror augmentation (p=0.5) + prompt paraphrases

Actions are scaled to [-1, 1] by the car's physical limits (``ActionScaler``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .recording import IMAGE_KEY, load_table
from .world import sample_episode

ACTION_COLS = ["action.vx", "action.vy", "action.wz"]
STATE_COLS = ["observation.state.vx", "observation.state.vy", "observation.state.wz"]
# Mirroring the image left<->right flips the sign of everything "sideways" or "turning":
#   vy (+ = left), wz (+ = counter-clockwise) and the goal bearing (+ = goal on the left). vx is unchanged.
MIRROR_SIGN = np.array([1.0, -1.0, -1.0], dtype=np.float32)


# -------------------------------------------------------------------------------------------------
# Action / state scaling
# -------------------------------------------------------------------------------------------------
@dataclass
class ActionScaler:
    """Divide (vx, vy, wz) by the car's speed limits -> every component lives in [-1, 1].

    Why not the usual mean/std normalisation? (1) Our limits are known exactly, so [-1, 1] is
    guaranteed; (2) vy is almost always 0, so its std is tiny and dividing by it would blow the rare
    strafing commands up to values of 5-10; (3) dividing by a constant keeps 0 at 0 and is
    symmetric, so a mirrored action stays a valid action."""

    limits: np.ndarray  # (3,) = (max_vx, max_vy, max_wz)

    def scale(self, a):
        return a / torch.as_tensor(self.limits, dtype=a.dtype, device=a.device) if torch.is_tensor(a) else a / self.limits

    def unscale(self, a):
        return a * torch.as_tensor(self.limits, dtype=a.dtype, device=a.device) if torch.is_tensor(a) else a * self.limits

    @classmethod
    def from_car(cls, car_cfg) -> "ActionScaler":
        return cls(np.array([car_cfg.max_vx, car_cfg.max_vy, car_cfg.max_wz], dtype=np.float32))


# -------------------------------------------------------------------------------------------------
# Language: paraphrases of "go to the <colour> <shape>"
# -------------------------------------------------------------------------------------------------
SHAPE_SYNONYMS = {
    "sphere": ["sphere", "ball", "orb"],
    "cube": ["cube", "block", "box"],
    "cylinder": ["cylinder", "pillar", "column"],
}
# Verb phrases seen in training. A separate set is kept for evaluation only, to test whether the model
# understands *new* wordings rather than memorising the training ones.
TRAIN_VERBS = ["drive to", "go to", "find", "navigate to", "head to", "move to", "go find", "get to",
               "approach", "make your way to", "reach", "travel to", "go over to", "roll to",
               "head over to", "drive up to", "steer to", "locate and drive to"]
EVAL_VERBS = ["proceed to", "seek out", "track down", "make a beeline for", "pull up next to"]
TRAIN_FRAMES = ["{v} the {c} {s}", "{v} the {c} {s}", "{v} the {c} {s}", "please {v} the {c} {s}",
                "{v} the {s} that is {c}", "can you {v} the {c} {s}", "{v} the {c} {s} and stop there",
                "your goal: the {c} {s}", "the {c} {s} - {v} it"]
EVAL_FRAMES = ["i want you to {v} the {c} {s}", "{v} that {c} {s}"]


class Paraphraser:
    """Random rewordings of a goal. Always mentions both colour and shape (never ambiguous)."""

    def __init__(self, split: str = "train", seed: int | None = None):
        self.verbs = TRAIN_VERBS if split == "train" else EVAL_VERBS
        self.frames = TRAIN_FRAMES if split == "train" else EVAL_FRAMES
        self.rng = np.random.default_rng(seed)

    def __call__(self, colour: str, shape: str) -> str:
        r = self.rng
        frame = self.frames[r.integers(len(self.frames))]
        v = self.verbs[r.integers(len(self.verbs))]
        if frame.startswith("the ") and frame.endswith("{v} it"):
            v = v.replace(" and drive to", "")
        text = frame.format(v=v, c=colour, s=SHAPE_SYNONYMS[shape][r.integers(len(SHAPE_SYNONYMS[shape]))])
        if r.random() < 0.15:
            text = text[0].upper() + text[1:]
        if r.random() < 0.15:
            text += "."
        return text


# -------------------------------------------------------------------------------------------------
# Frame cache + episode table
# -------------------------------------------------------------------------------------------------
def build_frame_cache(root: Path, repo_id: str, cache_path: Path, batch: int = 256, expected_frames: int | None = None) -> np.ndarray:
    """Decode every camera frame of the dataset ONCE into a uint8 array (N, H, W, 3) saved as .npy.

    Decoding video is slow for random access (each frame needs its key-frame and neighbours), and a
    training loop reads frames in random order millions of times. 5-6k frames x 196 KB = ~1.1 GB fits
    easily in RAM, so we pay the decoding cost once."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        if expected_frames is None or len(cached) == expected_frames:
            return cached
        del cached
        cache_path.unlink()  # the dataset grew (new recording session): decode again
    ds = LeRobotDataset(repo_id, root=root)
    out = np.lib.format.open_memmap(cache_path.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8,
                                    shape=(ds.num_frames, *ds.meta.features[IMAGE_KEY]["shape"]))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=4, shuffle=False)
    i = 0
    for b in loader:
        imgs = (b[IMAGE_KEY].permute(0, 2, 3, 1) * 255).round().to(torch.uint8).numpy()
        idx = b["index"].numpy()
        assert (idx == np.arange(i, i + len(idx))).all()
        out[i:i + len(idx)] = imgs
        i += len(idx)
    out.flush()
    del out
    cache_path.with_suffix(".tmp.npy").rename(cache_path)
    return np.load(cache_path, mmap_mode="r")


def episode_table(table) -> "pd.DataFrame":
    """One row per episode: frame range, seed, goal, whether the goal was visible at the start, etc."""
    import pandas as pd

    rows = []
    for e, g in table.groupby("episode_index"):
        spec = sample_episode(int(g["sim.episode_seed"].iloc[0]))
        shape, colour = spec.goal[:2]
        rows.append({"episode_index": int(e), "start": int(g["index"].min()), "stop": int(g["index"].max()) + 1,
                     "n_frames": len(g), "seed": spec.seed, "goal": f"{colour} {shape}", "colour": colour,
                     "shape": shape, "prompt": g.task.iloc[0], "start_visible": bool(spec.start_visible),
                     "collision_steps": int(g["sim.collision"].sum())})
    return pd.DataFrame(rows)


HELD_OUT_GOALS = ("orange sphere",)  # never a training goal: a test of combining words learned separately


# -------------------------------------------------------------------------------------------------
# The training dataset
# -------------------------------------------------------------------------------------------------
class NavChunkDataset(torch.utils.data.Dataset):
    """Sample i = one frame of one of the selected episodes, with the next ``chunk`` actions.

    Returns a dict:
      image        float32 (3, H, W) in [0, 1]  (smolVLA itself resizes to 512 and maps to [-1, 1])
      state        float32 (3,)   measured (vx, vy, wz), scaled by the speed limits
      action       float32 (chunk, 3)  actions t .. t+chunk-1, scaled to [-1, 1]
      action_pad   bool    (chunk,)   True where the chunk runs past the episode's end (action = 0 there)
      task         str      the prompt (a random paraphrase if ``paraphrase``)
      aux          float32 (4,)   privileged targets: sin(bearing), cos(bearing), surface distance, goal visible
      mirrored     bool
    """

    def __init__(self, table, frames: np.ndarray, episodes, episode_ids, scaler: ActionScaler, chunk: int = 10,
                 mirror_p: float = 0.0, paraphrase: bool = False, seed: int = 0):
        self.frames, self.scaler, self.chunk, self.mirror_p = frames, scaler, chunk, mirror_p
        eps = episodes.set_index("episode_index").loc[list(episode_ids)]
        self.goal = {e: (r["colour"], r["shape"]) for e, r in eps.iterrows()}  # r.shape would be the row size!
        self.prompt = {e: r.prompt for e, r in eps.iterrows()}
        sel = table[table.episode_index.isin(set(episode_ids))]
        self.index = sel["index"].to_numpy()  # global frame index (row in the frame cache)
        self.ep_of = sel.episode_index.to_numpy()
        self.ep_stop = eps.loc[self.ep_of, "stop"].to_numpy()  # first global index AFTER the episode
        tab = table.set_index("index")
        self.actions = scaler.scale(tab.loc[:, ACTION_COLS].to_numpy(np.float32))  # indexed by global index
        self.states = scaler.scale(tab.loc[:, STATE_COLS].to_numpy(np.float32))
        b = tab["sim.goal_rel.bearing"].to_numpy(np.float32)
        self.aux = np.stack([np.sin(b), np.cos(b), tab["sim.goal_rel.surface_distance"].to_numpy(np.float32),
                             tab["sim.goal_visible"].to_numpy(np.float32)], axis=1)
        self.paraphraser = Paraphraser("train", seed) if paraphrase else None
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        g, e, stop = int(self.index[i]), int(self.ep_of[i]), int(self.ep_stop[i])
        n = min(self.chunk, stop - g)
        act = np.zeros((self.chunk, 3), np.float32)
        act[:n] = self.actions[g:g + n]  # past the episode's end the car stands still at the goal -> 0
        pad = np.arange(self.chunk) >= n
        img = np.array(self.frames[g])  # copy: the cache is a read-only memory map
        state, aux = self.states[g].copy(), self.aux[g].copy()
        mirrored = self.rng.random() < self.mirror_p
        if mirrored:
            img = img[:, ::-1]
            act, state = act * MIRROR_SIGN, state * MIRROR_SIGN
            aux[0] = -aux[0]  # sin(bearing) flips, cos(bearing) doesn't
        task = self.paraphraser(*self.goal[e]) if self.paraphraser else self.prompt[e]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0,
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(act),
            "action_pad": torch.from_numpy(pad),
            "task": task,
            "aux": torch.from_numpy(aux),
            "mirrored": mirrored,
            "episode_index": e,
            "frame": g,
        }

    def worker_init(self, worker_id):
        """Give every DataLoader worker its own random stream (otherwise all workers mirror identically)."""
        seed = torch.initial_seed() % 2**32
        self.rng = np.random.default_rng(seed)
        if self.paraphraser:
            self.paraphraser.rng = np.random.default_rng(seed + 1)


def save_prep(path: Path, prep: dict) -> None:
    Path(path).write_text(json.dumps(prep, indent=2))


def load_prep(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def load_training_data(prep_path: Path):
    """Everything training needs, from the file notebook 03 writes: (prep, table, frames, episodes).
    A combined prep (with a "datasets" list) gives one table/frame array over all its datasets; the table
    then has a "source" column (dataset name) and globally unique frame and episode numbers."""
    prep = load_prep(prep_path)
    table = combined_table(prep["datasets"]) if "datasets" in prep else load_table(prep["dataset_root"])
    frames = np.load(prep["frame_cache"], mmap_mode="r")
    assert len(frames) == len(table), "frame cache out of date - re-run notebook 03"
    return prep, table, frames, episode_table(table)


# -------------------------------------------------------------------------------------------------
# Several recordings as one dataset (the six scripted folders)
# -------------------------------------------------------------------------------------------------
def combined_table(parts: list[dict]):
    """Tables of several datasets stacked. Frame numbers ("index") and episode numbers are shifted so they
    stay unique: dataset 2 starts where dataset 1 ends. The first dataset keeps its numbers, so splits
    made for it alone stay valid. parts = [{"name", "root", ...}, ...]."""
    import pandas as pd

    tabs, f_off, e_off = [], 0, 0
    for p in parts:
        t = load_table(p["root"]).copy()
        t["index"] += f_off
        t["episode_index"] += e_off
        t["source"] = p.get("source", p["name"])   # several recorder folders can form one source
        f_off, e_off = f_off + len(t), int(t["episode_index"].max()) + 1
        tabs.append(t)
    return pd.concat(tabs, ignore_index=True)


def build_combined_cache(parts: list[dict], cache_path: Path) -> np.ndarray:
    """One frame array for all datasets, in the same order as ``combined_table`` (rebuilt if out of date)."""
    caches = [build_frame_cache(p["root"], p["repo_id"], p["frame_cache"], expected_frames=len(load_table(p["root"])))
              for p in parts]
    n = sum(len(c) for c in caches)
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = np.load(cache_path, mmap_mode="r")
        if len(cached) == n:
            return cached
        del cached
        cache_path.unlink()
    out = np.lib.format.open_memmap(cache_path.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint8, shape=(n, *caches[0].shape[1:]))
    i = 0
    for c in caches:
        out[i:i + len(c)] = c
        i += len(c)
    out.flush()
    del out
    cache_path.with_suffix(".tmp.npy").rename(cache_path)
    return np.load(cache_path, mmap_mode="r")


def split_by_layout(episodes, frac_val: float = 0.15, held_out_goals=HELD_OUT_GOALS, seed: int = 0) -> dict:
    """Train/val split by layout. In the words scenes the two halves of a pair (same layout, other object asked) share
    the same picture, so they go to the SAME side - otherwise validation would contain pictures seen in training."""
    rng = np.random.default_rng(seed)
    ep = episodes[~episodes.goal.isin(held_out_goals)]
    layout = ep.seed.to_numpy() // 2
    groups = np.unique(layout)
    n_val = max(1, int(round(frac_val * len(groups)))) if len(groups) > 1 else 0
    val_groups = set(rng.choice(groups, size=n_val, replace=False).tolist())
    val = sorted(int(e) for e, g in zip(ep.episode_index, layout) if g in val_groups)
    train = sorted(int(e) for e in ep.episode_index if e not in val)
    excluded = sorted(int(e) for e in episodes.episode_index if e not in set(train) | set(val))
    return {"train": train, "val": val, "excluded": excluded}


def make_scripted_prep(folders, out_path, source: str = "expert_close", frac_val: float = 0.10, base_prep=None):
    """Training file for scripted (expert) recordings, optionally on top of an existing prep (e.g. human demos).

    folders: recorder output folders (each a LeRobot dataset); they form one source `source`.
    Scripted drives are split by LAYOUT (frac_val of layouts to validation, never trained on).
    base_prep: None = scripted only; or a prep path whose datasets and split are kept unchanged.
    Returns the prep dict (also saved to out_path) and prints drives/frames per source."""
    from pathlib import Path
    root = Path(out_path).resolve().parents[2]   # runs/datasets/X.json -> project root
    base = load_prep(base_prep) if base_prep else None
    parts = (list(base["datasets"]) if base else []) + [
        {"name": Path(f).name, "source": source, "root": str(Path(f).resolve()), "repo_id": f"local/{Path(f).name}",
         "frame_cache": str(root / "runs/cache" / f"{Path(f).name}_frames.npy")} for f in sorted(folders)]
    table = combined_table(parts)
    cache = root / "runs/cache" / f"{Path(out_path).stem}_frames.npy"
    build_combined_cache(parts, cache)
    eps = episode_table(table)
    eps["source"] = eps.episode_index.map(table.groupby("episode_index").source.first())
    sc = split_by_layout(eps[eps.source == source], frac_val=frac_val, seed=0)
    if base:
        split = {k: sorted(base["split"][k] + sc[k]) for k in ("train", "val", "excluded")}
        split["held_out_goals"] = base["split"]["held_out_goals"]
        prep = {**base}
    else:
        split = sc
        from .config import CarConfig
        prep = {"chunk": 10, "mirror_p": 0.0, "paraphrase": True,
                "action_limits": ActionScaler.from_car(CarConfig()).limits.tolist()}
    prep.update(datasets=parts, frame_cache=str(cache), split=split,
                n_train_frames=int(table.episode_index.isin(split["train"]).sum()),
                n_val_frames=int(table.episode_index.isin(split["val"]).sum()))
    save_prep(Path(out_path), prep)
    for s, g in eps.groupby("source"):
        tr = set(split["train"])
        print(f"{s:22s} drives {len(g):4d}  frames {int(table.source.eq(s).sum()):6d}  train drives {sum(e in tr for e in g.episode_index)}")
    return prep
