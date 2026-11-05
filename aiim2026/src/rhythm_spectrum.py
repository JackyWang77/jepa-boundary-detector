"""Rhythm spectrum gate (plan v2 section 3.3, preregistered first gate).

Compares event-interval distributions:
  PushT: push_onset inter-onset intervals, 100 episodes sampled seed=0
  cube : manip_onset inter-onset intervals, all 500 episodes
Metrics: variance, CV, log-dispersion (var of log-intervals), tail ratio
(p95/median), Brown-Forsythe (Levene center=median) on log-intervals.

Preregistered verdict:
  PASS    if cube log-dispersion > PushT AND BF p < 0.05 (cube more irregular)
  MARGINAL if only CV_cube > CV_pusht
  FAIL    otherwise -> stop, report, do not run cube E1/E6.
Outputs runs/rhythm_spectrum.{json,png}. Raw-frame intervals primary;
model-step (=raw/5, FS=5 both tasks) reported alongside.
"""
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from scipy import stats as sps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from fit_features import kinematic_events, ogb_cube_events

PUSHT_H5 = r'D:\stable_worldmodel\datasets\pusht_expert_train.h5'
CUBE_H5 = r'D:\stable_worldmodel\datasets\ogbench\cube_single_expert.h5'
RUNS = SRC.parent / 'runs'
FS = 5


def inter_onset_intervals(event_lists):
    out = []
    for on in event_lists:
        idx = np.nonzero(on)[0]
        if len(idx) >= 2:
            out.append(np.diff(idx))
    return (np.concatenate(out) if out else np.array([])).astype(float)


def describe(x):
    if len(x) == 0:
        return {}
    return {
        'n': int(len(x)),
        'mean': float(np.mean(x)), 'median': float(np.median(x)),
        'std': float(np.std(x)), 'var': float(np.var(x)),
        'cv': float(np.std(x) / np.mean(x)) if np.mean(x) > 0 else None,
        'p95_over_median': (float(np.percentile(x, 95) / np.median(x))
                            if np.median(x) > 0 else None),
        'log_var': float(np.var(np.log(np.maximum(x, 1.0)))),
    }


def main():
    # ---- PushT events (100 eps, seed 0)
    with h5py.File(PUSHT_H5, 'r') as f:
        state = f['state'][:]
        ep_len = f['ep_len'][:]
        ep_off = f['ep_offset'][:]
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(len(ep_len), 100, replace=False))
    pusht_lists = [kinematic_events(state[ep_off[i]:ep_off[i] + ep_len[i]])[0]
                   for i in sample]
    n_events_p = [int(x.sum()) for x in pusht_lists]
    iv_p = inter_onset_intervals(pusht_lists)

    # ---- cube events (all eps)
    with h5py.File(CUBE_H5, 'r') as f:
        pos = f['privileged_block_0_pos'][:]
        ep_len_c = f['ep_len'][:]
        ep_off_c = f['ep_offset'][:]
    cube_lists = [ogb_cube_events(pos[ep_off_c[i]:ep_off_c[i] + ep_len_c[i]])[0]
                  for i in range(len(ep_len_c))]
    n_events_c = [int(x.sum()) for x in cube_lists]
    iv_c = inter_onset_intervals(cube_lists)

    # ---- stats + gate
    dp, dc = describe(iv_p), describe(iv_c)
    logp = np.log(np.maximum(iv_p, 1.0))
    logc = np.log(np.maximum(iv_c, 1.0))
    bf = sps.levene(logp, logc, center='median')  # Brown-Forsythe
    var_ratio = dc['log_var'] / dp['log_var'] if dp['log_var'] > 0 else None
    cv_gt = (dc['cv'] is not None and dp['cv'] is not None
             and dc['cv'] > dp['cv'])
    disp_gt = dc['log_var'] > dp['log_var']
    if disp_gt and bf.pvalue < 0.05:
        verdict = 'PASS'
    elif cv_gt:
        verdict = 'MARGINAL'
    else:
        verdict = 'FAIL'

    out = {
        'pusht': {'eps': 100, 'events_per_ep_median':
                  float(np.median(n_events_p)), **dp},
        'cube': {'eps': 500, 'events_per_ep_median':
                 float(np.median(n_events_c)), **dc},
        'log_dispersion_ratio_cube_over_pusht': var_ratio,
        'brown_forsythe_on_log_intervals': {
            'stat': float(bf.statistic), 'p': float(bf.pvalue)},
        'cv_gt': bool(cv_gt), 'dispersion_gt': bool(disp_gt),
        'units': 'raw frames (model step = raw/5, FS=5 both tasks)',
        'verdict': verdict,
    }
    RUNS.mkdir(exist_ok=True)
    (RUNS / 'rhythm_spectrum.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    # ---- plot
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(0, np.log10(max(iv_p.max(), iv_c.max())), 40)
    ax.hist(iv_p, bins=bins, alpha=0.55, label=f'PushT (n={len(iv_p)})',
            color='#3d6dd6')
    ax.hist(iv_c, bins=bins, alpha=0.55, label=f'cube (n={len(iv_c)})',
            color='#c07b3d')
    ax.set_xscale('log')
    ax.set_xlabel('inter-onset interval (raw frames)')
    ax.set_ylabel('count')
    ax.set_title(f'rhythm spectrum — verdict: {verdict} '
                 f'(log-dispersion ratio {var_ratio:.2f}, BF p='
                 f'{bf.pvalue:.2e})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUNS / 'rhythm_spectrum.png', dpi=130)
    print(f'-> {RUNS / "rhythm_spectrum.png"}')


if __name__ == '__main__':
    main()
