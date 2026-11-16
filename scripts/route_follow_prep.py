"""Training file for notebook 06's route-following drives (runs/datasets/expert_pf_*), with the SAME validation and
excluded layouts as notebook 03's split (runs/datasets/scripted_close_prep.json), so results stay comparable.

usage: python scripts/route_follow_prep.py   -> runs/datasets/route_follow_prep.json"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from carsim.data import load_prep, load_training_data, make_scripted_prep, save_prep
    D = ROOT / "runs/datasets"
    old = load_prep(D / "scripted_close_prep.json")
    _, _ot, _, oe = load_training_data(D / "scripted_close_prep.json")
    seed_of = dict(zip(oe.episode_index, oe.seed))
    old_val = {seed_of[e] for e in old["split"]["val"]}; old_exc = {seed_of[e] for e in old["split"]["excluded"]}
    folders = sorted(str(p) for p in D.glob("expert_pf_*") if p.is_dir())
    prep = make_scripted_prep(folders, str(D / "route_follow_prep.json"), source="expert_route")
    _, t, _, e = load_training_data(D / "route_follow_prep.json")
    split = {"train": [], "val": [], "excluded": [], "held_out_goals": old["split"].get("held_out_goals", ["orange sphere"])}
    for ep, sd in zip(e.episode_index, e.seed):
        split["val" if sd in old_val else "excluded" if sd in old_exc else "train"].append(int(ep))
    prep["split"] = split
    prep["n_train_frames"] = int(t.episode_index.isin(split["train"]).sum()); prep["n_val_frames"] = int(t.episode_index.isin(split["val"]).sum())
    save_prep(D / "route_follow_prep.json", prep)
    print({k: len(v) for k, v in split.items() if k != "held_out_goals"}, "frames train/val:", prep["n_train_frames"], prep["n_val_frames"])
