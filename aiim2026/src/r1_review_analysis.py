"""R1 reviewer analysis (mac, local data only).

Answers reviewer concern 1: do the three signals fire spuriously on ordinary
motion, and how many deployed fires are false?

Part A: signal distributions at kinematic event steps vs non-event steps
        (raw f1, f2, f3_dual over all 500 PushT training episodes).
Part B: deployed dual-head (head_r2_dual_cnn.pt) over the recorded hold-out
        split: fire precision / recall / false fires per episode, using the
        exact kinematic-event labels and deploy normalization.
Part C: same stats for the Cube model head from cube_r2_dual.json, and
        real-post fire counts from cube probe replan_logs.

Outputs:
  results/r1_review.json
  Templates/Conference-LaTeX-template_10-17-19/fig3_signal_dist.pdf
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
from e4_train_head import CausalCNN, apply_refractory
from fit_features import kinematic_events
from r2_dualhead import dual_innov

ROOT = Path(__file__).resolve().parents[1]
FEAT, MODELS, RESULTS = ROOT / 'features', ROOT / 'models', ROOT / 'results'
TEXDIR = ROOT.parent / 'Templates' / 'Conference-LaTeX-template_10-17-19'
H5 = Path.home() / '.stable_worldmodel/datasets/pusht_expert_train.h5'
FS = 5
WINDOWS = (2, 4, 8)
CTX = 3
TAU, REFRACT, MATCH_TOL = 0.5, 5, 2

BLUE, GRAY, ORANGE = '#0072BD', '#8C8C8C', '#D95319'


def build_episode(ep, f5, W):
    """Replicate r2_dualhead.load_episode: 18ch X, event steps, K."""
    fd = np.load(FEAT / f'ep{ep}.npz')
    K = int(fd['model_len'])
    T = int(fd['raw_len'])
    off = int(f5['ep_offset'][ep])
    s = f5['state'][off:off + T]
    push, settle = kinematic_events(s)
    ev = np.unique(-(-np.nonzero(push | settle)[0] // FS))
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
    return X.astype(np.float32), ev, K, fd['f1'], fd['f2'], f3d


def main():
    f5 = h5py.File(H5, 'r')
    W = np.load(MODELS / 'fz_dual_linear.npy')
    meta = json.load(open(MODELS / 'r2_meta_dual.json'))
    mu = np.array(meta['feat_mu'], np.float32)
    sd = np.array(meta['feat_sd'], np.float32)

    hold_ids = meta['hold_episode_ids']
    head = CausalCNN(in_ch=18)
    head.load_state_dict(torch.load(MODELS / meta['head_ckpt'],
                                    map_location='cpu'))
    head.eval()

    # ---- Part B: hold-out fire audit with the deployed head
    tp = fp = fn = 0
    fires_per_ep, eps_len = [], []
    with torch.no_grad():
        for ep in hold_ids:
            X, ev, K, *_ = build_episode(ep, f5, W)
            x = (torch.from_numpy(X) - torch.from_numpy(mu)) \
                / torch.from_numpy(sd)
            p = torch.sigmoid(head(x[None]))[0].numpy()
            pred = np.nonzero(apply_refractory(p > TAU))[0] + 3
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
            eps_len.append(K)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    part_b = {
        'hold_episodes': len(hold_ids),
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'fires_per_episode': round(float(np.mean(fires_per_ep)), 3),
        'false_fires_per_episode': round(fp / len(hold_ids), 3),
    }

    # ---- Part A: signal distributions (all recorded episodes)
    eps_all = json.load(open(FEAT / 'meta.json'))['episode_ids']
    f1_ev, f1_ne, f2_ev, f2_ne, f3_ev, f3_ne = [], [], [], [], [], []
    for ep in eps_all:
        X, ev, K, f1r, f2r, f3r = build_episode(ep, f5, W)
        mask_ev = np.isin(np.arange(3, K), ev)
        mask_ne = ~mask_ev
        f1_ev.append(f1r[3:][mask_ev]); f1_ne.append(f1r[3:][mask_ne])
        f2_ev.append(f2r[3:][mask_ev]); f2_ne.append(f2r[3:][mask_ne])
        f3_ev.append(f3r[3:][mask_ev]); f3_ne.append(f3r[3:][mask_ne])

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
    fig.savefig(TEXDIR / 'fig3_signal_dist.pdf', bbox_inches='tight')
    plt.close(fig)

    # ---- Part C: cube numbers from recorded artifacts
    cube = json.load(open(RESULTS / 'cube_r2_dual.json'))
    best = max(cube['sweep'], key=lambda r: r['fire_micro_f1'])
    ctp, cfp, cfn = best['tp_fp_fn']
    part_c = {
        'cube_head': best['head'],
        'cube_hold_auc': best['hold_auc'],
        'cube_f1': best['fire_micro_f1'],
        'cube_precision': round(ctp / max(ctp + cfp, 1), 4),
        'cube_false_fires': cfp,
    }
    cube_fires = []
    for jf in sorted((RESULTS / 'cube_probe_raw').glob('*d75*')):
        d = json.load(open(jf))
        for _, v in d['replan_logs'].items():
            if isinstance(v, list):
                cube_fires.append(len(v))
    part_c['cube_real_fires_per_ep'] = round(float(np.mean(cube_fires)), 3)

    out = {'part_b_pushT_holdout': part_b,
           'part_c_cube': part_c}
    json.dump(out, open(RESULTS / 'r1_review.json', 'w'), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
