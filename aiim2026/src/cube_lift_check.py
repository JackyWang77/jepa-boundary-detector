"""Does the oracle expert ever LIFT the cube? -> pick-place vs push.

Reads qpos from the local cube h5, extracts the block free-joint z
(qpos adr 14: x,y,z,quat), and reports per-episode lift statistics
relative to each episode's resting height.
"""
import json
from pathlib import Path

import h5py
import numpy as np

H5 = Path(r'D:\stable_worldmodel\datasets\ogbench\cube_single_expert.h5')
ADR = 14          # object_joint_0 qpos address (validated in patch_cube_h5)
LIFT = 0.03       # 3 cm above resting = unambiguous lift (cube is 2 cm)

with h5py.File(H5, 'r') as f:
    qpos = f['qpos'][:]          # (N, nq)
    z = qpos[:, ADR + 2]
    ep = f['episode_idx'][:]

eps = np.unique(ep)
max_lift, lift_steps, total_steps = [], 0, 0
for e in eps:
    ze = z[ep == e]
    rest = np.median(ze[:5])
    d = ze - rest
    max_lift.append(d.max())
    lift_steps += int((d > LIFT).sum())
    total_steps += len(ze)

max_lift = np.array(max_lift)
stats = {
    'n_episodes': int(len(eps)),
    'resting_z_median_overall': float(np.median([np.median(z[ep == e][:5])
                                                 for e in eps])),
    'max_lift_per_ep': {
        'median': float(np.median(max_lift)),
        'p10': float(np.percentile(max_lift, 10)),
        'p90': float(np.percentile(max_lift, 90)),
        'max': float(max_lift.max()),
    },
    'episodes_with_lift_gt_3cm': int((max_lift > LIFT).sum()),
    'lifted_step_share': lift_steps / total_steps,
}
print(json.dumps(stats, indent=2))
