"""E3: unsupervised change-point segmentation on the E2 feature streams.

Per EXPERIMENT_GUIDE.md: PELT primary (cost rbf or normal, BIC penalty, no
PushT-specific tuning); controls BinSeg / BottomUp / BOCPD on the same
stream. PELT changepoints become the E4 pseudo-labels.

Signal: the 6 E2 stat channels {f1,f2,f3} x {mean,std}, trimmed to the
common valid head (index >= 3), corpus z-scored over the 500 episodes.
Penalty (BIC):  normal cost -> 0.5 * (d + d(d+1)/2) * log(n)
                 rbf cost   -> d * log(n)   (piecewise-constant-mean analog)
BOCPD: Adams-MacKay, constant hazard 1/L with L = median PELT segment
length (granularity-matched control), obs var 1, prior N(0,1) per channel.
Output: aiim2026/segments/e3/{ep_id}.npz + meta.json (*_smoke for --smoke).
Changepoint convention: b = first step of the new segment (global model
step index, including the 3-step head offset).
"""

import json
import time
from pathlib import Path

import numpy as np
import ruptures as rpt

FEAT = Path(__file__).resolve().parents[1] / 'features'
OUT = Path(__file__).resolve().parents[1] / 'segments' / 'e3'
CHANNELS = ['f1_mean', 'f1_std', 'f2_mean', 'f2_std', 'f3_mean', 'f3_std']
HEAD = 3          # first index where all 6 channels are defined
D = len(CHANNELS)


def load_signals(ep_ids, suffix=''):
    sigs, meta = [], []
    for ep in ep_ids:
        d = np.load(FEAT / f'ep{ep}{suffix}.npz')
        K = int(d['model_len'])
        sig = np.stack([d[c] for c in CHANNELS], axis=1)[HEAD:]
        assert not np.isnan(sig).any(), f'NaN in signal ep{ep}'
        sigs.append(sig)
        meta.append((int(ep), K))
    return sigs, meta


def bic_pen(model, n):
    if model == 'normal':
        p = D + D * (D + 1) // 2        # mean + full covariance params
        return 0.5 * p * np.log(n)
    # piecewise-constant-mean analog for kernel costs
    return D * np.log(n)


def segment_ruptures(sig, model, pen):
    algo = rpt.Pelt(model=model).fit(sig)
    cps_local = algo.predict(pen=pen)
    return np.array(cps_local[:-1], dtype=int)   # drop the final segment end


def segment_binseg(sig, model, pen):
    algo = rpt.Binseg(model=model).fit(sig)
    return np.array(algo.predict(pen=pen)[:-1], dtype=int)


def segment_bottomup(sig, model, pen):
    algo = rpt.BottomUp(model=model).fit(sig)
    return np.array(algo.predict(pen=pen)[:-1], dtype=int)


def bocpd_exact(sig, hazard):
    """Exact small-T AMcay: keep per-run-length sufficient statistics."""
    T = len(sig)
    D_ = sig.shape[1]
    logH, log1mH = np.log(hazard), np.log(1 - hazard)
    # R[r] over r = 0..T at time t (log probs), Ssum[r] per-channel sums
    logR = np.zeros(T + 1)
    logR[1:] = -np.inf                  # at t=0 run length 0 has prob 1
    Ssum = np.zeros((T + 1, D_))
    count = np.zeros(T + 1)             # number of obs in run r
    map_run = np.zeros(T, dtype=int)
    cps = []
    for t in range(T):
        x = sig[t]
        # predictive logpdf per run length: x ~ N(m_r, 1 + v_r)
        v_r = 1.0 / (count + 1.0)
        m_r = v_r[:, None] * Ssum       # (T+1, D)
        var = 1.0 + v_r
        ll = -0.5 * (np.log(2 * np.pi * var[:, None])
                     + (x - m_r) ** 2 / var[:, None]).sum(axis=1)
        # growth + reset
        with np.errstate(divide='ignore'):
            growth = logR + ll + log1mH
            reset = np.logaddexp.reduce(logR + ll) + logH
        newR = np.full(T + 1, -np.inf)
        newR[1:] = growth[:-1]           # run length r -> r+1
        newR[0] = reset
        # normalize
        norm = np.logaddexp.reduce(newR[np.isfinite(newR)])
        logR = newR - norm
        # update sufficient stats
        Ssum = np.vstack([np.zeros((1, D_)), Ssum[:-1] + x])
        count = np.concatenate([[0], count[:-1] + 1])
        map_run[t] = int(np.argmax(logR))
        if t > 0 and map_run[t] < map_run[t - 1]:
            cps.append(t)
    return np.array(cps, dtype=int)


def main(smoke=False):
    suffix = '_smoke' if smoke else ''
    OUT.mkdir(parents=True, exist_ok=True)
    meta_feats = json.load(open(FEAT / 'meta.json'))
    ep_ids = meta_feats['episode_ids']
    if smoke:
        ep_ids = ep_ids[:2]

    sigs, eps_meta = load_signals(ep_ids)
    corpus = np.concatenate(sigs)
    mu, sd = corpus.mean(0), corpus.std(0) + 1e-8
    sigs = [(s - mu) / sd for s in sigs]

    # PELT cost: BIC penalty requires a likelihood cost, so 'normal' is the
    # principled choice ('rbf' is kernel-based and degenerated to 0 cps in
    # smoke under the same penalty). Documented, not tuned.
    model = 'normal'

    t0 = time.time()
    all_cps = {m: [] for m in ('pelt', 'binseg', 'bottomup', 'bocpd')}
    seg_lens = []
    for sig, (ep, K) in zip(sigs, eps_meta):
        n = len(sig)
        pen = bic_pen(model, n)
        cps = {
            'pelt': segment_ruptures(sig, model, pen),
            'binseg': segment_binseg(sig, model, pen),
            'bottomup': segment_bottomup(sig, model, pen),
        }
        all_cps['pelt'].append(cps['pelt'])
        if len(cps['pelt']):
            seg_lens.extend(np.diff(np.concatenate([[-1], cps['pelt']])))
        all_cps['binseg'].append(cps['binseg'])
        all_cps['bottomup'].append(cps['bottomup'])

    L = float(np.median(seg_lens)) if seg_lens else float(
        np.mean([len(s) for s in sigs]))
    hazard = 1.0 / max(L, 1.0)
    for sig, (ep, K) in zip(sigs, eps_meta):
        cps = bocpd_exact(sig, hazard)
        all_cps['bocpd'].append(cps)

    for (ep, K), i in zip(eps_meta, range(len(eps_meta))):
        out = {}
        for m in ('pelt', 'binseg', 'bottomup', 'bocpd'):
            # local -> global model-step index: cp = end of segment -> the
            # first step of the new segment is cp + HEAD
            out[f'cps_{m}'] = all_cps[m][i] + HEAD
        labels = np.zeros(K, dtype=np.int8)
        for b in out['cps_pelt']:
            if 0 <= b < K:
                labels[b] = 1
        out['labels_pelt'] = labels
        np.savez_compressed(OUT / f'ep{ep}{suffix}.npz',
                            ep_id=ep, model_len=K, **out)

    meta = {
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'experiment': 'E3 change-point segmentation',
        'signal': CHANNELS,
        'head_offset': HEAD,
        'corpus_mean': mu.tolist(),
        'corpus_std': sd.tolist(),
        'pelt_cost': model,
        'penalty': 'BIC (see bic_pen)',
        'bocpd': {'hazard': hazard, 'median_pelt_seg_len': L,
                  'obs_var': 1.0, 'prior': 'N(0,1) per channel'},
        'n_episodes': len(ep_ids),
        'smoke': smoke,
        'cp_convention': 'b = first model step of the new segment',
    }
    with open(OUT / f'meta{suffix}.json', 'w') as f:
        json.dump(meta, f, indent=2)

    for m in all_cps:
        counts = [len(c) for c in all_cps[m]]
        print(f'{m}: cps/episode median={np.median(counts):.0f} '
              f'mean={np.mean(counts):.1f} min={min(counts)} max={max(counts)} '
              f'total={sum(counts)}')
    print(f'done {len(ep_ids)} episodes in {time.time() - t0:.1f}s -> {OUT}')


if __name__ == '__main__':
    import sys
    main(smoke='--smoke' in sys.argv)
