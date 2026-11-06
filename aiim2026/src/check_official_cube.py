"""Audit the official OGBench cube-single-play npz before rebuilding the eval
dataset from it.

Checks the two contaminations that killed the locally collected h5:
  (1) state sanity   -- qpos/qvel width + block block_0 pose slice (ADR=14)
  (2) start/goal gap -- distance from a sampled start to its +offs goal,
                        i.e. how often an episode starts inside the 4cm
                        success radius (the "null baseline" bug)
Also reports episode layout (terminals -> lengths) so we know we can cut
episodes without re-collecting.
"""
import json
from pathlib import Path

import numpy as np

NPZ = Path(r'D:\stable_worldmodel\datasets\ogbench\official'
           r'\cube-single-play-v0.npz')
OUT = Path(r'd:\Research-2026-jepa-boundary-detector\aiim2026\runs'
           r'\official_cube_audit.json')
ADR = 14  # object_joint_0 qpos address (patch_cube_h5.py, model.nq=21)


def main():
    f = np.load(NPZ, allow_pickle=True)
    qpos = f['qpos']
    qvel = f['qvel']
    act = f['actions']
    term = f['terminals'].astype(bool)
    n = len(qpos)
    print(f'rows={n}  qpos={qpos.shape}  qvel={qvel.shape} '
          f'act={act.shape}  terminals={term.sum()}')

    # --- episode layout
    ep_end = np.nonzero(term)[0]
    ep_len = np.diff(np.concatenate([[-1], ep_end]))
    print(f'episodes={len(ep_len)}  len: min={ep_len.min()} '
          f'median={np.median(ep_len):.0f} max={ep_len.max()}  '
          f'mean={ep_len.mean():.1f}')

    # --- block pose from qpos
    pos = qpos[:, ADR:ADR + 3].astype(np.float64)
    quat = qpos[:, ADR + 3:ADR + 7].astype(np.float64)
    qn = np.linalg.norm(quat, axis=1)
    print(f'block pos range x[{pos[:,0].min():.3f},{pos[:,0].max():.3f}] '
          f'y[{pos[:,1].min():.3f},{pos[:,1].max():.3f}] '
          f'z[{pos[:,2].min():.3f},{pos[:,2].max():.3f}]  '
          f'quat norm [{qn.min():.4f},{qn.max():.4f}]')

    # --- start -> goal gap, per offset, restricted to playable starts
    rng = np.random.default_rng(0)
    res = {'rows': int(n), 'episodes': int(len(ep_len)),
           'ep_len_median': float(np.median(ep_len)),
           'ep_len_mean': float(ep_len.mean()),
           'offsets': {}}
    ep_starts = np.concatenate([[0], ep_end[:-1] + 1])
    lens = ep_len
    for off in (25, 50, 75):
        max_start = lens - off - 1
        ok_eps = np.nonzero(max_start > 0)[0]
        k = min(2000, len(ok_eps))
        pick = rng.choice(ok_eps, size=k, replace=False)
        starts = ep_starts[pick] + rng.integers(
            0, max_start[pick], size=k)
        goals = starts + off
        d = np.linalg.norm(pos[starts] - pos[goals], axis=1)
        res['offsets'][str(off)] = {
            'n': int(k),
            'median_m': float(np.median(d)),
            'p10_m': float(np.percentile(d, 10)),
            'frac_within_4cm': float((d <= 0.04).mean()),
            'frac_within_2cm': float((d <= 0.02).mean()),
        }
        print(f'offset={off:>2}  median gap={np.median(d):.3f}m  '
              f'p10={np.percentile(d,10):.3f}m  '
              f'<4cm={100*(d<=0.04).mean():.1f}%  '
              f'<2cm={100*(d<=0.02).mean():.1f}%')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, 'w'), indent=2)
    print('->', OUT)


if __name__ == '__main__':
    main()
