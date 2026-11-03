"""Eyeball-check the kinematic event ground truth.

For 5 hold episodes: print event tables, plot the decision signals
(agent-block distance vs r_c, block displacement vs threshold) with event
markers, and dump raw image strips around sampled events for visual
cross-checking. Outputs: runs/event_check.png, runs/event_frames.png.
"""

from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_features import kinematic_events, BLOCK_DISP

RUNS = Path(__file__).resolve().parents[1] / 'runs'
H5 = Path.home() / '.stable_worldmodel/datasets/pusht_expert_train.h5'
N_TOTAL, N_TRAIN, SEED = 400, 300, 0
N_SHOW = 5


def hold_episode_ids():
    """Reconstruct the episode ids chosen by extract_latents.py (same seed)."""
    with h5py.File(H5, 'r', swmr=True) as f:
        n_eps = len(f['ep_len'][:])
    rng = np.random.default_rng(SEED)
    eps = np.sort(rng.choice(n_eps, N_TOTAL, replace=False))
    return eps[N_TRAIN:]          # hold split


def main():
    d = np.load(RUNS / 'latents_hold.npz')
    states, ep_len = d['state'], d['ep_len']
    ep_ids = hold_episode_ids()

    # split hold states into episodes
    eps_state = []
    off = 0
    for n in ep_len:
        eps_state.append(states[off:off + int(n)])
        off += int(n)

    # ---- console tables + signal plots --------------------------------
    fig, axes = plt.subplots(N_SHOW, 2, figsize=(12, 3 * N_SHOW))
    for i in range(N_SHOW):
        s = eps_state[i]
        push, settle = kinematic_events(s)
        dist = np.linalg.norm(s[:, 0:2] - s[:, 2:4], axis=1)
        disp = np.concatenate([[0], np.linalg.norm(np.diff(s[:, 2:4], axis=0), axis=1)])
        T = len(s)

        print(f'\n=== hold episode #{i} (orig ep {ep_ids[i]}, T={T}) ===')
        print(f'{"frame":>6} {"dist(t)":>9} {"blk_disp":>9}  event')
        for t in range(1, T):
            ev = []
            if push[t]:
                ev.append(f'PUSH_ONSET (disp {disp[max(0,t-3):t+3].mean():.2f}px/step)')
            if settle[t]:
                ev.append('SETTLE')
            if ev:
                print(f'{t:>6} {dist[t]:>9.1f} {disp[t]:>9.2f}  ' + '; '.join(ev))

        ax = axes[i, 0]
        ax.plot(dist, lw=0.9, color='#333')
        for t in np.nonzero(push)[0]:
            ax.axvline(t, color='#B2182B', lw=2)
        for t in np.nonzero(settle)[0]:
            ax.axvline(t, color='#1B7837', lw=2)
        ax.set_ylabel('agent-block dist', fontsize=9)
        ax.set_title(f'hold ep #{i}  (push onset red / settle green)', fontsize=9)
        ax.tick_params(labelsize=7)

        ax = axes[i, 1]
        ax.plot(disp, lw=0.9, color='#333')
        ax.axhline(BLOCK_DISP, color='#1B7837', ls='--', lw=1,
                   label=f'th={BLOCK_DISP}px')
        for t in np.nonzero(push)[0]:
            ax.axvline(t, color='#B2182B', lw=2)
        ax.set_ylabel('block disp/step', fontsize=9)
        ax.set_title('block displacement + push onsets', fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(RUNS / 'event_check.png', dpi=150)
    print('\nsaved -> runs/event_check.png')

    # ---- raw frame strips around sampled events ------------------------
    picks = []
    for i in range(N_SHOW):
        s = eps_state[i]
        push, settle = kinematic_events(s)
        for ev_arr, name in [(push, 'PUSH_ONSET'), (settle, 'SETTLE')]:
            ts = np.nonzero(ev_arr)[0]
            if len(ts):
                picks.append((i, name, int(ts[0])))
    picks = picks[:6]           # max 6 strips

    with h5py.File(H5, 'r', swmr=True) as f:
        ep_len_all, ep_off_all = f['ep_len'][:], f['ep_offset'][:]
        fig, axes = plt.subplots(len(picks), 5, figsize=(13, 2.4 * len(picks)))
        for r, (i, name, t) in enumerate(picks):
            ep = ep_ids[i]
            off = int(ep_off_all[ep])
            span = range(t - 4, t + 1)          # frames t-4 .. t (event at t)
            for c, ft in enumerate(span):
                ax = axes[r, c]
                ax.imshow(f['pixels'][off + ft])
                ax.set_title(f't={ft}', fontsize=8)
                ax.axis('off')
            axes[r, 0].set_ylabel(f'ep#{i}\n{name}', fontsize=8)
        fig.suptitle('event at t: check the frame where the event fires '
                     '(rightmost column)', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(RUNS / 'event_frames.png', dpi=120)
    print('saved -> runs/event_frames.png')


if __name__ == '__main__':
    main()
