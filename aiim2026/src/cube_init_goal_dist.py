"""How far is the goal at episode start, per eval (d, seed)?

Replicates e6_eval's sampling (episode_len / valid mask / rng(seed).choice)
on qpos-derived block positions, then reports:
  - init->goal Euclidean distance distribution
  - share of episodes already inside the 4 cm success ball at t=0
That share is a lower bound on success for a policy that never moves the
cube (the null policy), contextualizing lane C's 61-77% numbers.
"""
import json

import h5py
import numpy as np

H5 = r'D:\stable_worldmodel\datasets\ogbench\cube_single_expert.h5'
ADR = 14
SEEDS = (42, 43, 44)
DS = (25, 50, 75)
N_EVAL = 50
BALL = 0.04

with h5py.File(H5, 'r') as f:
    qpos = f['qpos'][:, ADR:ADR + 3]
    ep_col = f['episode_idx'][:]
    step_col = f['step_idx'][:]

ep_indices, _ = np.unique(ep_col, return_index=True)
ep_len = np.array([step_col[ep_col == ep].max() + 1 for ep in ep_indices])
max_start = ep_len - 1  # per-episode cap, refined per d below

out = {}
for d in DS:
    per_seed = {}
    for seed in SEEDS:
        cap = ep_len - d - 1
        cap_by_ep = dict(zip(ep_indices, cap))
        cap_rows = np.array([cap_by_ep[e] for e in ep_col])
        valid = np.nonzero(step_col <= cap_rows)[0]
        rng = np.random.default_rng(seed)
        pick = np.sort(valid[rng.choice(len(valid), size=N_EVAL,
                                        replace=False)])
        init_pos = qpos[pick]
        goal_pos = qpos[pick + d]
        dist = np.linalg.norm(init_pos - goal_pos, axis=1)
        per_seed[seed] = {
            'dist_median_m': float(np.median(dist)),
            'dist_p10': float(np.percentile(dist, 10)),
            'dist_p90': float(np.percentile(dist, 90)),
            'already_in_4cm_ball_pct': float(100 * (dist <= BALL).mean()),
        }
    agg = {k: float(np.mean([per_seed[s][k] for s in SEEDS]))
           for k in next(iter(per_seed.values()))}
    out[f'd{d}'] = {'per_seed': per_seed, 'mean': agg}

print(json.dumps(out, indent=2))
with open(r'..\runs\cube_init_goal_dist.json', 'w') as f:
    json.dump(out, f, indent=2)
