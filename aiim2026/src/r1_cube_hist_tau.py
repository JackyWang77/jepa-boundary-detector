"""R1 reviewer evidence on Cube: signal distributions + tau sweep (4060 side).

Mirrors `r1_review_analysis.py` Part A / Part D on the Cube task. No training,
no evaluation, CPU only: reads features_cube/*.npz and the deployed Cube head.

Part A: f1 / f2 / f3_dual densities at kinematic (manip onset | settle) steps
        vs non-event steps over all extracted Cube episodes
        -> results/fig3_cube_dist.pdf
Part D: hold-out tau sweep {0.30..0.70 step 0.05}, refractory=5, match tol=2,
        gated exactly like the mac-side PushT audit
        -> results/r1_cube_hist_tau.json
"""

import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_extract_features import causal_stats
from e4_train_head import TinyGRU, apply_refractory
from fit_features import ogb_cube_events
from r2_dualhead import dual_innov, WINDOWS

ROOT = Path(__file__).resolve().parents[1]
FEAT, MODELS, RESULTS = ROOT / 'features_cube', ROOT / 'models', ROOT / 'results'
H5 = Path(r'D:\stable_worldmodel\datasets\cube_single_expert.h5')
FS = 5
MATCH_TOL = 2

BLUE, GRAY = '#0072BD', '#8C8C8C'


def build_episode(ep, f5, W):
    """Cube twin of r1_review_analysis.build_episode (18ch X + raw signals)."""
    fd = np.load(FEAT / f'ep{ep}.npz')
    K = int(fd['model_len'])
    T = int(fd['raw_len'])
    off = int(f5['ep_offset'][ep])
    pos = f5['privileged_block_0_pos'][off:off + T]
    onset, settle = ogb_cube_events(pos)
    ev = np.unique(-(-np.nonzero(onset | settle)[0] // FS))
    ev = ev[ev < K]

    f3d = dual_innov(fd['z'], W)
    chans = []
    for name in ('f1', 'f2'):
        raw = fd[name]
        for w in WINDOWS:
            m, sd = causal_stats(raw, w)
            chans.extend([m, sd])
    for w in WINDOWS:
        m, sd = causal_stats(f3d, w)
        chans.extend([m, sd])
    X = np.stack(chans, axis=1)[3:]
    return (X.astype(np.float32), ev, K, fd['f1'], fd['f2'], f3d)


def main():
    meta = json.loads((FEAT / 'meta.json').read_text())
    eps_all = meta['episode_ids']
    W = np.load(MODELS / 'cube_fz_dual_linear.npy')
    dmeta = json.loads((MODELS / 'cube_r2_meta_dual.json').read_text())
    mu = np.array(dmeta['feat_mu'], np.float32)
    sd = np.array(dmeta['feat_sd'], np.float32)
    hold_ids = dmeta['hold_episode_ids']
    # same checkpoint the cube probe arm deploys (selected in cube_r2_dual.json)
    head = TinyGRU(in_ch=18)
    head.load_state_dict(torch.load(MODELS / 'cube_head_r2_dual_gru.pt',
                                    map_location='cpu'))
    head.eval()
    print(f'cube: {len(eps_all)} episodes, hold {len(hold_ids)}, '
          f'windows {WINDOWS}')

    hold_preds = []
    f1_ev, f1_ne, f2_ev, f2_ne, f3_ev, f3_ne = [], [], [], [], [], []
    with h5py.File(H5, 'r', swmr=True) as f5, torch.no_grad():
        for ep in eps_all:
            X, ev, K, f1r, f2r, f3r = build_episode(ep, f5, W)
            idx = np.arange(3, K)
            mask_ev = np.isin(idx, ev)
            mask_ne = ~mask_ev
            for raw, lst_ev, lst_ne in ((f1r, f1_ev, f1_ne),
                                        (f2r, f2_ev, f2_ne),
                                        (f3r, f3_ev, f3_ne)):
                v = raw[3:]
                v = v[np.isfinite(v)]
                if v.shape[0] != idx.shape[0]:
                    # trailing NaN (f3_dual at t=K-1): align by dropping tail
                    idx_v = idx[np.isfinite(raw[3:])]
                    m_ev = np.isin(idx_v, ev)
                else:
                    m_ev = mask_ev
                lst_ev.append(v[m_ev])
                lst_ne.append(v[~m_ev])

            if ep in hold_ids:
                x = (torch.from_numpy(X) - torch.from_numpy(mu)) \
                    / torch.from_numpy(sd)
                p = torch.sigmoid(head(x[None]))[0].numpy()
                hold_preds.append((p, ev, K))

    # ---- Part D: tau sweep on the recorded hold-out (mac-side audit logic)
    def audit(tau):
        tp = fp = fn = 0
        fires_per_ep = []
        for p, ev, K in hold_preds:
            pred = np.nonzero(apply_refractory(p > tau))[0] + 3
            matched, t = set(), 0
            for q in pred:
                near = [e for e in ev
                        if abs(int(q) - int(e)) <= MATCH_TOL and e not in matched]
                if near:
                    matched.add(near[0])
                    t += 1
            tp += t
            fp += len(pred) - t
            fn += len(ev) - t
            fires_per_ep.append(len(pred))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        return {'tau': round(float(tau), 2),
                'tp': tp, 'fp': fp, 'fn': fn,
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'fires_per_episode': round(float(np.mean(fires_per_ep)), 3),
                'false_fires_per_episode': round(fp / len(hold_preds), 3)}

    part_d = [audit(t) for t in np.arange(0.30, 0.71, 0.05)]
    part_b = {'hold_episodes': len(hold_preds)}
    part_b.update({k: v for k, v in audit(0.5).items() if k != 'tau'})

    out = {'hold_episodes': len(hold_preds),
           'part_b_cube_holdout': part_b,
           'part_d_tau_sweep_cube': part_d}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'r1_cube_hist_tau.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(part_b, indent=2))
    for r in part_d:
        print(f"  tau={r['tau']:.2f} P={r['precision']:.3f} "
              f"R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"fire/ep={r['fires_per_episode']:.2f} "
              f"false/ep={r['false_fires_per_episode']:.2f}")

    # ---- Part A: identical figure style to the PushT panel
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for ax, (ev, ne, title) in zip(
            axes,
            [(np.concatenate(f1_ev), np.concatenate(f1_ne), r'$f_1$ energy'),
             (np.concatenate(f2_ev), np.concatenate(f2_ne), r'$f_2$ cosine'),
             (np.concatenate(f3_ev), np.concatenate(f3_ne), r'$f_3$ novelty')]):
        ax.hist(ne, bins=60, density=True, histtype='step', lw=1.4,
                color=GRAY, label='non-event')
        ax.hist(ev, bins=60, density=True, histtype='step', lw=1.4,
                color=BLUE, label='event')
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
    axes[0].set_ylabel('density', fontsize=7)
    axes[0].legend(fontsize=7, loc='upper right', frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(RESULTS / 'fig3_cube_dist.pdf', bbox_inches='tight')
    plt.close(fig)
    print('wrote', RESULTS / 'fig3_cube_dist.pdf')


if __name__ == '__main__':
    main()
