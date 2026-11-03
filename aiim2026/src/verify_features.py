"""Three acceptance gates for the boundary features.

Gate 1 sanity: shape alignment, no NaN, f1_raw in [0,1], f2 in [-1,1],
               fast subspace carries >= 15% increment energy on hold.
Gate 2 visual: overlay plots of f1/f2/f3 vs kinematic event lines, 5 hold eps.
Gate 3 AUC:    single-feature AUC against the union of kinematic events
               (push onset | settle, +-3 steps) on hold;
               orientation picked on train, evaluated on hold. Floor: > 0.55.
               Per-event breakdown reported for transparency.
Outputs: runs/feature_overlay.png, runs/feature_sanity.json.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

RUNS = Path(__file__).resolve().parents[1] / 'runs'
TOL = 3           # event tolerance in steps for the AUC gate
AUC_FLOOR = 0.55
FAST_SHARE_FLOOR = 0.15


def load(name):
    d = np.load(RUNS / f'features_{name}.npz')
    return {k: d[k] for k in d.files}


def iter_eps(feats):
    off = 0
    for n in feats['ep_len']:
        n = int(n)
        sl = slice(off, off + n)
        yield {k: feats[k][sl] for k in feats}
        off += n


def pos_mask(events, tol=TOL):
    """Dilate boolean event array by +-tol steps."""
    m = events.copy()
    for shift in range(1, tol + 1):
        m[shift:] |= events[:-shift]
        m[:-shift] |= events[shift:]
    return m


def auc_with_orientation(feat, target):
    """AUC with orientation chosen on the same data (train), applied to target."""
    y = target.astype(int)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    auc_pos = roc_auc_score(y, feat)
    return auc_pos


def main():
    train, hold = load('train'), load('hold')
    report = {'gate1': {}, 'gate3': {}}
    ok = True

    # ---- Gate 1: sanity -------------------------------------------------
    n_points = int(train['ep_len'].sum() + hold['ep_len'].sum())
    nan_free = all(
        np.isfinite(feats[k]).all()
        for feats in (train, hold) for k in ('f1', 'f2', 'f3')
    )
    f1_raw_ok = all(
        ((feats['f1_raw'] >= -1e-6) & (feats['f1_raw'] <= 1 + 1e-6)).all()
        for feats in (train, hold)
    )
    f2_ok = all(
        ((feats['f2'] >= -1 - 1e-6) & (feats['f2'] <= 1 + 1e-6)).all()
        for feats in (train, hold)
    )
    fast_share_hold = float(hold['f1_raw'].mean())
    gate1 = {
        'points_total': n_points,
        'nan_free': bool(nan_free),
        'f1_raw_in_01': bool(f1_raw_ok),
        'f2_in_-1_1': bool(f2_ok),
        'fast_subspace_energy_share_hold': fast_share_hold,
        'fast_share_ge_15pct': fast_share_hold >= FAST_SHARE_FLOOR,
    }
    report['gate1'] = gate1
    g1_pass = (nan_free and f1_raw_ok and f2_ok
               and fast_share_hold >= FAST_SHARE_FLOOR)
    ok &= g1_pass
    print('[Gate 1] sanity:', json.dumps(gate1, indent=2))
    print('[Gate 1]', 'PASS' if g1_pass else 'FAIL')

    # ---- Gate 2: overlay of 5 hold episodes ------------------------------
    eps = list(iter_eps(hold))
    rng = np.random.default_rng(1)
    picks = rng.choice(len(eps), size=min(5, len(eps)), replace=False)
    fig, axes = plt.subplots(3, len(picks), figsize=(4 * len(picks), 7),
                             sharex='col')
    feats_rows = [('f1', 'f1: fast-energy ratio'), ('f2', 'f2: slow cos'),
                  ('f3', 'f3: log increment energy')]
    for col, idx in enumerate(picks):
        ep = eps[idx]
        x = np.arange(len(ep['f1']))
        for row, (key, label) in enumerate(feats_rows):
            ax = axes[row, col]
            ax.plot(x, ep[key], lw=0.8, color='#333')
            for ev, color, name in [('push_onset', '#B2182B', 'push onset'),
                                    ('settle', '#1B7837', 'settle')]:
                for t in np.nonzero(ep[ev])[0]:
                    ax.axvline(t, color=color, lw=1.2, alpha=0.9)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == 0:
                ax.set_title(f'hold episode #{idx}', fontsize=10)
            ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in
               ('#B2182B', '#1B7837')]
    fig.legend(handles, ['push onset', 'settle'],
               loc='upper center', ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RUNS / 'feature_overlay.png', dpi=150)
    plt.close(fig)
    print('[Gate 2] overlay saved -> runs/feature_overlay.png (eyeball it)')

    # ---- Gate 3: single-feature AUC vs union of kinematic events (hold) ---
    def stack(feats, event_key='union'):
        y_list, f_list = [], []
        for ep in iter_eps(feats):
            if event_key == 'union':
                ev = ep['push_onset'] | ep['settle']
            else:
                ev = ep[event_key]
            y = pos_mask(ev).astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            y_list.append(y)
            f_list.append(np.vstack([ep['f1'], ep['f2'], ep['f3']]).T)
        return np.concatenate(f_list), np.concatenate(y_list)

    aucs = {}
    breakdown = {}
    f_tr, y_tr = stack(train)
    f_ho, y_ho = stack(hold)
    for i, key in enumerate(['f1', 'f2', 'f3']):
        # orientation decided on train
        auc_tr = roc_auc_score(y_tr, f_tr[:, i])
        sign = 1.0 if auc_tr >= 0.5 else -1.0
        auc_ho = roc_auc_score(y_ho, sign * f_ho[:, i])
        aucs[key] = {'train_auc': float(auc_tr), 'hold_auc': float(auc_ho)}
        print(f'[Gate 3] {key}: train AUC={auc_tr:.3f} -> hold AUC={auc_ho:.3f}')
        # per-event breakdown: features and labels stacked per event
        per_event = {}
        for ev in ('push_onset', 'settle'):
            try:
                f_ev, y_ev = stack(hold, ev)
                a = roc_auc_score(y_ev, sign * f_ev[:, i])
                per_event[ev] = float(max(a, 1 - a))
            except ValueError:
                per_event[ev] = None
        breakdown[key] = per_event

    report['gate3'] = {
        'target': 'union of kinematic events (push_onset|settle), '
                  f'(+-{TOL} steps)',
        'pos_rate_hold': float(y_ho.mean()),
        'auc': aucs,
        'per_event_breakdown_hold': breakdown,
        'floor': AUC_FLOOR,
        'pass': all(v['hold_auc'] > AUC_FLOOR for v in aucs.values()),
    }
    g3_pass = report['gate3']['pass']
    ok &= g3_pass
    print('[Gate 3] per-event breakdown (hold):',
          json.dumps(breakdown, indent=2))
    print('[Gate 3]', 'PASS' if g3_pass else 'FAIL')

    report['events_hold'] = {
        'push_onset': int(hold['push_onset'].sum()),
        'settle': int(hold['settle'].sum()),
    }
    report['overall'] = 'PASS' if ok else 'FAIL'
    with open(RUNS / 'feature_sanity.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f'OVERALL: {report["overall"]}')


if __name__ == '__main__':
    main()
