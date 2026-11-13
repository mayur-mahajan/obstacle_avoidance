"""Does the reward rank behaviours sensibly? Runs a few episodes each of: the autopilot, standing still,
random driving. Run as a separate process (from notebook 05) so its simulator's memory is freed after."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from carsim.env import NavEnv
from carsim.expert import PathFollowingExpert
from carsim.rl_env import RewardTracker, random_spec

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
env = NavEnv(backend="cpu")
rng = np.random.default_rng(5)
for name in ("autopilot", "stand still", "random"):
    rows = []
    for i in range(N):
        spec = random_spec(np.random.default_rng(100 + i))
        obs, info = env.reset(spec=spec)
        tr, ex, ret = RewardTracker(env, info), PathFollowingExpert(env), 0.0
        while True:
            a = (ex.act() if name == "autopilot" else (0, 0, 0) if name == "stand still"
                 else (rng.uniform(-.3, .8), rng.uniform(-.6, .6), rng.uniform(-1.5, 1.5)))
            obs, term, trunc, info = env.step(a)
            ret += tr(info)
            if term or trunc:
                break
        rows.append((ret, info["success"], tr.totals))
    parts = {k: np.mean([r[2][k] for r in rows]) for k in rows[0][2]}
    print(f"{name:12s} score per drive {np.mean([r[0] for r in rows]):7.2f}   success {np.mean([r[1] for r in rows]):4.0%}   "
          + "  ".join(f"{k} {v:+.2f}" for k, v in parts.items()), flush=True)
