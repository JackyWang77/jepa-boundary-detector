"""Paper figures: Fig.2 drift curves (re-anchor vs none), Fig.3 T' histogram.

Outputs vector PDFs (+ PNG previews) into the LaTeX template directory.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import torch

from probe_rollout_fire import (CTX, ROOT, DEVICE, feat_on_seq, head_probs,
                                load_head, load_model, make_rows)

TEMPLATE = ROOT.parent / 'Templates' / 'Conference-LaTeX-template_10-17-19'
FEAT_DIR = ROOT / 'features'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.6,
})


def real_stream_fires(z, act, model, head, mu, sd, fz, tau=0.5, refractory=5):
    f1, f2, f3 = feat_on_seq(z, act, model, fz=fz)
    rows, valid_s = make_rows(f1, f2, f3, mu, sd)
    probs = np.array(head_probs(head, rows))
    fires, last = [], -10**9
    for s, p in zip(valid_s, probs):
        if p > tau and s - last >= refractory:
            fires.append(s)
            last = s
    return fires, valid_s, probs


def rollout_drift(z, act, model, fires=None):
    """Autoregressive rollout driven by real action blocks. If fires given,
    re-anchor (reset context to real latents) at each fire index.
    Returns drift array indexed by model step (nan before first prediction)."""
    K = len(z)
    z_t = [torch.from_numpy(z[i]).float().to(DEVICE) for i in range(K)]
    seq = [z_t[0], z_t[1], z_t[2]]
    drift = np.full(K, np.nan)
    fires = set(fires or [])
    for t in range(K - CTX):
        g = CTX + t                       # global index being predicted
        if g in fires:
            seq = [z_t[g - 2], z_t[g - 1], z_t[g]]   # re-anchor on real
            drift[g] = 0.0
            continue
        emb = torch.stack(seq[-CTX:])[None]
        act_w = torch.from_numpy(act[t:t + CTX]).float().to(DEVICE)[None]
        with torch.no_grad():
            act_emb = model.action_encoder(act_w)
            z_next = model.predict(emb, act_emb)[0, -1]
        seq.append(z_next)
        drift[g] = float(torch.norm(z_next - z_t[g]).item())
    return drift


def pick_episode(model, head, mu, sd, fz, hold_ids):
    best = None
    for eid in hold_ids:
        path = FEAT_DIR / f'ep{eid}.npz'
        if not path.exists():
            continue
        d = np.load(path)
        z, act = d['z'], d['action']
        if len(z) < 22:
            continue
        fires, _, _ = real_stream_fires(z, act, model, head, mu, sd, fz)
        fires = [s for s in fires if s >= CTX and s <= len(z) - 3]
        if 2 <= len(fires) <= 4:
            spread = np.diff(fires).min() if len(fires) > 1 else 99
            score = (spread, len(z))
            if best is None or score > best[0]:
                best = (score, eid, z, act, fires)
    if best is None:
        raise RuntimeError('no suitable episode with 2-4 fires found')
    return best[1], best[2], best[3], best[4]


def _box(ax, x, y, w, h, text, fc, fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.03,rounding_size=0.10',
                                fc=fc, ec='0.25', lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)


def _arr(ax, x1, y1, x2, y2, color='0.25', lw=0.9):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                shrinkA=1, shrinkB=1))


def fig1():
    """Pipeline schematic draft (to be redrawn in PPT by the user)."""
    GRAY, BLUE, ORANGE = '#EFEFEF', '#D6E9F8', '#FADBD2'
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.set_xlim(-0.7, 10.2)
    ax.set_ylim(0, 8.0)
    ax.axis('off')
    bh = 1.15

    # offline stage, snake layout: row1 left->right, row2 right->left
    _box(ax, 0.1, 6.6, 2.7, bh, 'frozen JEPA\nencoder', GRAY)
    _box(ax, 3.65, 6.6, 2.7, bh, 'latent\nstream $z_t$', GRAY)
    _box(ax, 7.2, 6.6, 2.7, bh, 'signals\n$f_1,f_2,f_3$', GRAY)
    _box(ax, 0.1, 4.6, 2.7, bh, 'causal head\n$p_t$ ($<$4k params)', BLUE)
    _box(ax, 3.65, 4.6, 2.7, bh, 'distill', GRAY)
    _box(ax, 7.2, 4.6, 2.7, bh, 'PELT\nsegmentation', GRAY)

    # online stage: two posts then planner
    _box(ax, 0.6, 2.4, 4.0, bh, 'imagined post\nprobe rollout $\\to$ horizon $h$', BLUE)
    _box(ax, 5.4, 2.4, 4.0, bh, 'real post\nfire $\\to$ truncate + re-anchor', ORANGE)
    _box(ax, 3.0, 0.3, 4.0, bh, 'CEM planner', BLUE)

    # arrows row 1
    _arr(ax, 2.8, 7.17, 3.65, 7.17)
    _arr(ax, 6.35, 7.17, 7.2, 7.17)
    # row1 -> row2 (snake) and row 2 right->left
    _arr(ax, 8.55, 6.6, 8.55, 5.75)
    _arr(ax, 7.2, 5.17, 6.35, 5.17)
    _arr(ax, 3.65, 5.17, 2.8, 5.17)
    # head -> both posts (elbow routing)
    ax.plot([1.45, 1.45], [4.6, 4.05], color='0.25', lw=0.9)
    ax.plot([1.45, 7.4], [4.05, 4.05], color='0.25', lw=0.9)
    _arr(ax, 1.45, 4.05, 1.45, 3.55)
    _arr(ax, 7.4, 4.05, 7.4, 3.55)
    # posts -> planner
    _arr(ax, 2.6, 2.4, 4.0, 1.45)
    _arr(ax, 7.4, 2.4, 6.0, 1.45)
    # executed actions / new observations back to the stream
    ax.annotate('', xy=(5.0, 6.6), xytext=(7.0, 0.87),
                arrowprops=dict(arrowstyle='-|>', color='0.55', lw=0.7,
                                connectionstyle='arc3,rad=-0.38',
                                shrinkA=1, shrinkB=1))
    ax.text(10.0, 3.6, 'actions,\nnew\nobservations', fontsize=5.8,
            color='0.4', ha='center', va='center')

    # stage labels
    ax.text(-0.45, 6.17, 'offline', rotation=90, ha='center', va='center',
            fontsize=6.4, color='0.35')
    ax.text(-0.45, 2.1, 'online', rotation=90, ha='center', va='center',
            fontsize=6.4, color='0.35')

    fig.tight_layout(pad=0.2)
    for ext in ('pdf', 'png'):
        fig.savefig(TEMPLATE / f'fig1_pipeline.{ext}', dpi=300)
    plt.close(fig)


def fig2():
    model = load_model()
    head, meta = load_head(ROOT / 'models' / 'r2_meta_dual.json')
    mu = np.asarray(meta['feat_mu'], np.float32)
    sd = np.asarray(meta['feat_sd'], np.float32)
    fz = str(ROOT / 'models' / 'fz_dual_linear.npy')
    eid, z, act, fires = pick_episode(model, head, mu, sd, fz,
                                      meta['hold_episode_ids'])
    print(f'Fig2 episode={eid} K={len(z)} fires={fires}')

    drift_none = rollout_drift(z, act, model)
    drift_re = rollout_drift(z, act, model, fires=fires)
    steps = np.arange(len(z))

    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    ax.plot(steps, drift_none, color='0.55', lw=1.2, label='no re-anchor')
    ax.plot(steps, drift_re, color='#0072BD', lw=1.4,
            label='re-anchor at fires')
    for i, s in enumerate(fires):
        ax.plot([s, s], [0, drift_none[s]], color='#D95319', ls='--', lw=0.8,
                alpha=0.8, label='detector fire' if i == 0 else None, zorder=3)
    ax.set_xlabel('model step $t$')
    ax.set_ylabel(r'Error $\|\hat{z}_t - z_t\|_2$')
    ax.set_xlim(-0.4, len(z) - 0.6)
    ax.set_xticks(range(0, len(z)))
    ax.tick_params(axis='x', labelsize=6)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22),
              ncol=3, handlelength=1.6, columnspacing=1.0)
    ax.spines[['top', 'right']].set_visible(False)

    # red dots at the re-anchor points on the blue curve
    for s in fires:
        ax.plot(s, drift_re[s], 'o', color='#D95319', ms=4, zorder=5)
    ax.text(fires[0] + 2.3, 0.3, 're-anchor', fontsize=7.5,
            color='#0072BD', ha='center', va='bottom')
    ax.text(fires[1] + 2.4, 0.15, 're-anchor', fontsize=7.5,
            color='#0072BD', ha='center', va='bottom')
    # gray drift values at the same steps (no re-anchor side)
    for s in fires:
        ax.plot(s, drift_none[s], 'o', color='0.55', ms=2.5, zorder=5)
        ax.text(s, drift_none[s] + 0.3, f'{drift_none[s]:.1f}', fontsize=8,
                color='0.5', ha='center', va='bottom')

    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(TEMPLATE / f'fig2_drift.{ext}', dpi=300,
                    bbox_inches='tight')
    plt.close(fig)


def fig3():
    # PushT probe horizons, d=50 and d=75 (3 seeds each)
    def load_T(dist):
        ts = []
        for seed in (42, 43, 44):
            matches = list((ROOT / 'runs').glob(f'probe_t_hr_d{dist}_s{seed}*.json'))
            d = json.load(open(matches[0]))
            for ep_log in d['replan_logs'].values():
                for entry in ep_log:
                    for p in entry['probes']:
                        ts.append(p['T'])
        # T' is clamped to [5,10]; one log row held the raw value
        return np.clip(np.asarray(ts), 5, 10)

    ts50 = load_T(50)
    ts75 = load_T(75)

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
    for ax, ts, dist in ((axes[0], ts50, '50'), (axes[1], ts75, '75')):
        vals, counts = np.unique(ts, return_counts=True)
        share = 100 * counts / len(ts)
        print(f'Fig3 d={dist} T distribution:', dict(zip(vals.tolist(), counts.tolist())),
              f'n={len(ts)}, top share={share.max():.1f}%')
        xs = np.arange(len(vals))
        ax.bar(xs, share, width=0.45,
               color=['#2E86DE' if i == 0 else '#F48D3E' for i in range(len(vals))],
               edgecolor='none')
        for x, s in zip(xs, share):
            ax.text(x, s + 1.0, f'{s:.0f}%', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(v) for v in vals])
        ax.set_xlim(-0.6, len(vals) - 0.4)
        ax.set_ylim(0, 62)
        ax.set_xlabel("$T'$ (model steps)")
        ax.set_title(f'$d=50$ ($n$={len(ts)})' if dist == '50'
                     else f'$d=75$ ($n$={len(ts)})', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('share of probes (%)')
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(TEMPLATE / f'fig3_tdist.{ext}', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['fig1', 'fig2', 'fig3']
    if 'fig1' in which:
        fig1()
    if 'fig2' in which:
        fig2()
    if 'fig3' in which:
        fig3()
