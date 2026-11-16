"""Notebook 06 step 2a: PPO with help in bursts (carsim/guided_ppo.py), or the plain notebook-05 PPO for a car
without guidance (the control, e.g. F).

usage: python scripts/train_guided_ppo.py --init runs/checkpoints/car06_Cn_s0.pt --run runs/rl/car06_Cn_ppo_s0 [--rounds 100]
       (a car without a *_guidance.pt file gets plain notebook-05 PPO)
       --resume RUN/latest.pt --rounds 200 --patience 30: continue a run past round 100 (the demo weight is at its final
       low value from round ~105, so PPO has the most freedom then) and stop early once the 32-room check's return has
       not set a new best for `patience` rounds"""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--route", action="store_true", help="notebook 07: real-route car, real server route help in practice")
    ap.add_argument("--resume", default=None, help="continue from RUN/latest.pt")
    ap.add_argument("--patience", type=int, default=None, help="stop when no new best check return for this many rounds")
    ap.add_argument("--min_rounds", type=int, default=100, help="never stop early before this round (the demo lesson fades until ~105)")
    ap.add_argument("--prep", default=str(ROOT / "runs/datasets/route_follow_prep.json"))
    a = ap.parse_args()
    guided = Path(a.init).with_name(Path(a.init).stem + "_guidance.pt").exists()
    if a.route:
        from carsim.route_ppo import RoutePPOConfig as Cfg, RoutePPOTrainer as Tr
    elif guided:
        from carsim.guided_ppo import GuidedPPOConfig as Cfg, GuidedPPOTrainer as Tr
    else:
        from carsim.ppo import PPOConfig as Cfg, PPOTrainer as Tr
    tr = Tr(Cfg(seed=a.seed), a.init, a.run, a.prep)
    if a.resume:
        tr.load(a.resume)
        print("resumed at round", tr.round, flush=True)
    print("route" if a.route else "guided" if guided else "plain", "PPO from", a.init, "->", a.run, flush=True)
    while tr.round < a.rounds:
        e = tr.train_round()
        keep = ("round", "success_last100", "crash_episodes_last100", "return_last100", "kl", "explained_variance", "bc_loss", "val_demo_loss", "collect_s", "learn_s")
        print(json.dumps({k: (round(e[k], 3) if isinstance(e[k], float) else e[k]) for k in keep if k in e}), flush=True)
        if tr.round % tr.cfg.eval_every == 0:
            print("EVAL", json.dumps([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()}
                                      for x in tr.evals if x["round"] == tr.round]), flush=True)
            if a.patience:
                std = [e for e in tr.evals if e.get("tag", "standard") == "standard"]
                best = max(std, key=lambda e: e["return"])["round"]
                if tr.round >= a.min_rounds and tr.round - best >= a.patience:
                    print(f"early stop: no new best since round {best}", flush=True)
                    break
    tr.save("latest"); tr.close()
    print("done", flush=True)
