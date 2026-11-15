"""Cube task stage figure: initial / mid / final frames + block trajectory.

Reads cube_single_expert.h5 (no model, no eval), picks one successful episode
and renders three frames with the block-height time series below. Outputs
results/fig_cube_stages.pdf (+ png).
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


def main(ep=314):
    with h5py.File(H5, 'r', swmr=True) as f:
        o = int(f['ep_offset'][ep])
        L = int(f['ep_len'][ep])
        pix = f['pixels'][o:o + L]                    # (L, 224, 224, 3)
        block = f['privileged_block_0_pos'][o:o + L]  # (L, 3)
        tgt = f['privileged_target_block_pos'][o:o + L]
        grip = f['proprio_gripper_contact'][o:o + L]   # (L, 1)

    stages = [('initial', 0), ('manipulation', L // 2), ('final', L - 1)]
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
        ax.set_title(label, fontsize=9, pad=3)
    # trajectory: block z (height) + gripper contact
    ax = fig.add_subplot(gs[1, :])
    z = block[:, 2]
    # normalize z to [0,1] within episode for readability
    z0, z1 = z.min(), z.max()
    zn = (z - z0) / (z1 - z0 + 1e-9)
    ax.plot(t_axis, zn, color='#0072BD', lw=1.2, label='block height')
    contact = grip[:, 0]
    ax.fill_between(t_axis, -0.05, -0.05 + 0.18 * contact,
                    color='#D95319', alpha=0.45, lw=0, label='gripper contact')
    for label, t in stages:
        ax.axvline(t, color='0.5', ls='--', lw=0.7)
    ax.set_xlabel('env step')
    ax.set_ylabel('block height\n(norm.)')
    ax.set_ylim(-0.18, 1.05)
    ax.set_xlim(0, L - 1)
    ax.tick_params(axis='y', labelsize=6)
    ax.legend(frameon=False, fontsize=6.5, loc='upper right',
              bbox_to_anchor=(1.0, 1.0), ncol=2)
    ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f'Cube expert episode {ep}', fontsize=9, y=0.995)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f'fig_cube_stages_ep{ep}'
    fig.savefig(f'{out}.pdf', bbox_inches='tight')
    fig.savefig(f'{out}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}.pdf (.png) — ep={ep} L={L} '
          f'z_range=[{z0:.3f},{z1:.3f}]')


def grid(eps=(100, 314, 777)):
    """3 episodes x 3 stages, no trajectory, for a compact overview panel."""
    fig, axes = plt.subplots(len(eps), 3, figsize=(6.6, 6.4))
    with h5py.File(H5, 'r', swmr=True) as f:
        for row, ep in enumerate(eps):
            o = int(f['ep_offset'][ep]); L = int(f['ep_len'][ep])
            pix = f['pixels'][o:o + L]
            for col, (label, t) in enumerate(
                    [('initial', 0), ('mid', L // 2), ('final', L - 1)]):
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
    fig.suptitle('Cube: initial / mid / final frames', fontsize=9, y=0.995)
    fig.subplots_adjust(hspace=0.04, wspace=0.04, top=0.96, bottom=0.02,
                        left=0.08, right=0.98)
    out = RESULTS / 'fig_cube_stages_grid'
    fig.savefig(f'{out}.pdf', bbox_inches='tight')
    fig.savefig(f'{out}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}.pdf (.png)')


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if args and args[0] == 'grid':
        grid()
    else:
        ep = int(args[0]) if args else 314
        main(ep)
