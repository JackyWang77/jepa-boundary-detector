"""Cube task stage figure: initial / lift / final frames + block trajectory.

Task is goal-conditioned relocation: move the block onto the target position
(success when ||block - target|| <= ~0.035). Episodes are fixed at 201 frames
and the manipulation is over by ~frame 65, so L//2 and L-1 look identical;
the "manipulation" frame must be the lift apex, not the midpoint.

Outputs results/fig_cube_stages_ep{ep}.pdf (+ png).
"""
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H5 = Path(r'D:\stable_worldmodel\datasets\cube_single_expert.h5')
RESULTS = ROOT / 'results'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.6,
})


def lift_frame(block):
    """First rising frame at >=85% of the peak block height.

    The exact apex is usually occluded by the gripper fingers; a frame a few
    steps earlier still shows the cube clear of the table and inside the
    gripper, which is what the panel is meant to display.
    """
    z = block[:, 2]
    zr = (z - z.min()) / (z.max() - z.min() + 1e-9)
    peak = int(np.argmax(z))
    cand = np.nonzero(zr >= 0.85)[0]
    cand = cand[cand <= peak]
    return int(cand[0]) if len(cand) else peak


def main(ep=314):
    with h5py.File(H5, 'r', swmr=True) as f:
        o = int(f['ep_offset'][ep])
        L = int(f['ep_len'][ep])
        pix = f['pixels'][o:o + L]                    # (L, 224, 224, 3)
        block = f['privileged_block_0_pos'][o:o + L]  # (L, 3)
        tgt = f['privileged_target_block_pos'][o:o + L]
        grip = f['proprio_gripper_contact'][o:o + L]   # (L, 1)
        suc = f['success'][o:o + L].astype(bool)       # (L,)

    # stage times must follow the real timeline: the block reaches the target
    # around frame 63-65 and the arm then retracts, so L//2 == "already done".
    t_lift = lift_frame(block)
    stages = [('initial', 0), ('lift / carry', t_lift), ('final', L - 1)]
    t_axis = np.arange(L)

    fig = plt.figure(figsize=(6.6, 3.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[4.4, 1.4],
                          hspace=0.22, wspace=0.05)
    for col, (label, t) in enumerate(stages):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(pix[t])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f'{label}  ($t$={t})', fontsize=8.5, pad=3)
    # trajectory: block height, distance to target, gripper contact
    ax = fig.add_subplot(gs[1, :])
    z = block[:, 2]
    # normalize z to [0,1] within episode for readability
    z0, z1 = z.min(), z.max()
    zn = (z - z0) / (z1 - z0 + 1e-9)
    d = np.linalg.norm(block - tgt, axis=1)
    dn = d / (d.max() + 1e-9)
    ax.plot(t_axis, zn, color='#0072BD', lw=1.2, label='block height')
    ax.plot(t_axis, dn, color='#7E2F8E', lw=1.2, label='dist. to target')
    contact = grip[:, 0]
    ax.fill_between(t_axis, -0.05, -0.05 + 0.18 * contact,
                    color='#D95319', alpha=0.45, lw=0, label='gripper contact')
    # success window: the task is already solved at ~frame 64; the second
    # grasp/lift that follows is part of the demo, not new task progress
    on = np.nonzero(suc)[0]
    if len(on):
        ax.axvspan(on[0], L - 1, color='#77AC30', alpha=0.10, lw=0)
        ax.text(on[0] + 2.5, 1.03, 'success', color='#4C7A1A', fontsize=6.5,
                ha='left', va='center')
    for label, t in stages:
        ax.axvline(t, color='0.5', ls='--', lw=0.7)
    ax.set_xlabel('env step')
    ax.set_ylabel('norm.')
    ax.set_ylim(-0.18, 1.12)
    ax.set_xlim(0, L - 1)
    ax.tick_params(axis='y', labelsize=6)
    ax.legend(frameon=False, fontsize=6.5, loc='upper center',
              bbox_to_anchor=(0.5, -0.66), ncol=3, handlelength=1.6,
              columnspacing=1.6)
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'Cube expert episode {ep}: goal-conditioned block relocation',
                 fontsize=9, y=0.995)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f'fig_cube_stages_ep{ep}'
    fig.savefig(f'{out}.pdf', bbox_inches='tight')
    fig.savefig(f'{out}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}.pdf (.png) — ep={ep} L={L} lift t={t_lift} '
          f'z_range=[{z0:.3f},{z1:.3f}] d0={d[0]:.3f} d_end={d[-1]:.3f}')


def grid(eps=(100, 314, 777)):
    """3 episodes x 3 stages, no trajectory, for a compact overview panel."""
    fig, axes = plt.subplots(len(eps), 3, figsize=(6.6, 6.4))
    with h5py.File(H5, 'r', swmr=True) as f:
        for row, ep in enumerate(eps):
            o = int(f['ep_offset'][ep]); L = int(f['ep_len'][ep])
            pix = f['pixels'][o:o + L]
            t_lift = lift_frame(f['privileged_block_0_pos'][o:o + L])
            for col, (label, t) in enumerate(
                    [('initial', 0), ('lift', t_lift), ('final', L - 1)]):
                ax = axes[row, col]
                ax.imshow(pix[t])
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if row == 0:
                    ax.set_title(label, fontsize=9, pad=3)
                if col == 0:
                    ax.set_ylabel(f'ep. {ep}', fontsize=8, rotation=90,
                                  labelpad=2)
    fig.suptitle('Cube: initial / lift / final frames', fontsize=9, y=0.995)
    fig.subplots_adjust(hspace=0.04, wspace=0.04, top=0.96, bottom=0.02,
                        left=0.08, right=0.98)
    out = RESULTS / 'fig_cube_stages_grid'
    fig.savefig(f'{out}.pdf', bbox_inches='tight')
    fig.savefig(f'{out}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}.pdf (.png)')


def task(ep=314):
    """Three frames only (initial / lift / final), no caption and no time
    series. Sized like the Fig.3 signal-distribution row: figsize=(7.0, 2.4).
    """
    with h5py.File(H5, 'r', swmr=True) as f:
        o = int(f['ep_offset'][ep])
        L = int(f['ep_len'][ep])
        pix = f['pixels'][o:o + L]
        block = f['privileged_block_0_pos'][o:o + L]

    t_lift = lift_frame(block)
    stages = [('initial', 0), ('lift / carry', t_lift), ('final', L - 1)]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for ax, (label, t) in zip(axes, stages):
        ax.imshow(pix[t])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f'{label}  ($t$={t})', fontsize=9, pad=2)
    fig.subplots_adjust(left=0.004, right=0.996, top=0.90, bottom=0.004,
                        wspace=0.02)
    out = RESULTS / 'fig_cube_task'
    fig.savefig(f'{out}.pdf', bbox_inches='tight', pad_inches=0.01)
    fig.savefig(f'{out}.png', dpi=200, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print(f'wrote {out}.pdf (.png) — ep={ep} lift t={t_lift}')


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if args and args[0] == 'grid':
        grid()
    elif args and args[0] == 'task':
        task(int(args[1]) if len(args) > 1 else 314)
    else:
        ep = int(args[0]) if args else 314
        main(ep)
