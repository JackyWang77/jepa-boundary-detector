"""Cube event extraction sanity check (plan v2 section 3.2).

Runs ogb_cube_events over all 500 local oracle episodes, writes:
  runs/cube_event_thresh.json  -- displacement distribution + chosen thresholds
  runs/cube_event_check.png    -- 2-episode overlay for eyeball inspection
  console: per-episode event count stats (feeds the rhythm gate next).
"""
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from fit_features import (ogb_cube_events, CUBE_DISP, CUBE_RUN,
                          CUBE_SETTLE_DISP, CUBE_SETTLE_RUN)

H5 = r'D:\stable_worldmodel\datasets\ogbench\cube_single_expert.h5'
RUNS = SRC.parent / 'runs'


def main():
    with h5py.File(H5, 'r') as f:
        pos_all = f['privileged_block_0_pos'][:]
        ep_len = f['ep_len'][:]
        ep_off = f['ep_offset'][:]

    # ---- displacement distribution (calibration record)
    disp = np.zeros(len(pos_all))
    for i in range(len(ep_len)):
        s, e = ep_off[i], ep_off[i] + ep_len[i]
        disp[s + 1:e] = np.linalg.norm(np.diff(pos_all[s:e], axis=0), axis=1)
    thresh = {
        'n_frames': int(len(disp)),
        'zero_frac': float((disp == 0).mean()),
        'percentiles': {str(p): float(np.percentile(disp, p))
                        for p in (50, 75, 90, 95, 99, 99.5, 99.9)},
        'moving_frac_by_threshold': {
            f'{th:.0e}': float((disp > th).mean())
            for th in (1e-5, 1e-4, 5e-4, 1e-3, 2e-3)},
        'chosen': {'CUBE_DISP': CUBE_DISP, 'CUBE_RUN': CUBE_RUN,
                   'CUBE_SETTLE_DISP': CUBE_SETTLE_DISP,
                   'CUBE_SETTLE_RUN': CUBE_SETTLE_RUN},
    }

    # ---- events over all episodes
    stats = []
    events = {}
    for i in range(len(ep_len)):
        s, e = ep_off[i], ep_off[i] + ep_len[i]
        on, st = ogb_cube_events(pos_all[s:e])
        events[i] = (on, st)
        stats.append(dict(ep=i, n_onset=int(on.sum()),
                          n_settle=int(st.sum()),
                          moving_frac=float((disp[s:e] > CUBE_DISP).mean())))

    n_on = np.array([x['n_onset'] for x in stats])
    n_st = np.array([x['n_settle'] for x in stats])
    no_event_eps = [x['ep'] for x in stats if x['n_onset'] == 0]
    print(f'episodes: {len(stats)} | manip_onset/ep: median '
          f'{np.median(n_on):.0f} min {n_on.min()} max {n_on.max()} | '
          f'settle/ep: median {np.median(n_st):.0f} | '
          f'eps with 0 onsets: {len(no_event_eps)}')
    thresh['per_episode'] = {
        'onset_median': float(np.median(n_on)),
        'onset_min': int(n_on.min()), 'onset_max': int(n_on.max()),
        'settle_median': float(np.median(n_st)),
        'n_eps_zero_onsets': len(no_event_eps),
        'zero_onset_eps_sample': no_event_eps[:20],
    }
    RUNS.mkdir(exist_ok=True)
    (RUNS / 'cube_event_thresh.json').write_text(
        json.dumps(thresh, indent=2))

    # ---- overlay plot: pick a busy ep and a sparse ep (both with events)
    busy = max(stats, key=lambda x: x['n_onset'])['ep']
    cand = [x['ep'] for x in stats if 2 <= x['n_onset'] <= 3]
    sparse = cand[len(cand) // 2] if cand else busy
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for row, ep in enumerate((busy, sparse)):
        s, e = ep_off[ep], ep_off[ep] + ep_len[ep]
        d = np.linalg.norm(
            np.diff(pos_all[s:e], axis=0), axis=1)
        d = np.concatenate([[0.0], d])
        ax = axes[row]
        ax.semilogy(np.maximum(d, 1e-9), lw=0.7, color='#3d6dd6',
                    label='block disp/frame (m)')
        ax.axhline(CUBE_DISP, color='#c04b58', ls='--', lw=1,
                   label=f'MANIP {CUBE_DISP:.0e}')
        ax.axhline(CUBE_SETTLE_DISP, color='#58c97e', ls=':', lw=1,
                   label=f'SETTLE {CUBE_SETTLE_DISP:.0e}')
        on, st = events[ep]
        ax.plot(np.nonzero(on)[0], [CUBE_DISP * 3] * int(on.sum()), 'rv',
                ms=7, label=f'manip_onset ({int(on.sum())})')
        ax.plot(np.nonzero(st)[0], [CUBE_SETTLE_DISP / 3] * int(st.sum()),
                'g^', ms=6, label=f'settle ({int(st.sum())})')
        ax.set_title(f'episode {ep} '
                     f'({"busy" if row == 0 else "sparse"})')
        ax.set_xlabel('raw frame')
        ax.legend(fontsize=8, ncol=3)
    fig.suptitle('ogb_cube_events: block-pose velocity threshold + debounce')
    fig.tight_layout()
    fig.savefig(RUNS / 'cube_event_check.png', dpi=130)
    print(f'-> {RUNS / "cube_event_thresh.json"} , '
          f'{RUNS / "cube_event_check.png"}')


if __name__ == '__main__':
    main()
