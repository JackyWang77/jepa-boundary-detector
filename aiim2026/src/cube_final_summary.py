"""Cube L1/L3 summary: per-arm success, paired McNemar, T' telemetry.

Reads runs/cube_flat_*_n100.json + runs/cube_probe_t_hr_*_n100.json and the
matching .log files (for probe replans/fires + T' counts).
"""
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
DS = [25, 50, 75]
SEEDS = [42, 43, 44]
HORIZONS = [(5, 5), (10, 10), (15, 15)]


def load(name):
    p = RUNS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def mcnemar(a, b):
    """Paired McNemar, continuity-corrected chi2(df=1). a,b bool arrays."""
    if len(a) == 0:
        return None
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        return 0.0, 1.0, n01, n10
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = math.erfc(math.sqrt(max(chi2, 0) / 2))
    return chi2, p, n01, n10


def arm_key(d, h=None, probe=False):
    if probe:
        return f'probe·d{d}'
    return f'flat({h[0]},{h[1]})·d{d}'


def main():
    flat = {}
    for d in DS:
        for h in HORIZONS:
            for s in SEEDS:
                j = load(f'cube_flat_h{h[0]}r{h[1]}_d{d}_s{s}_n100.json')
                if j:
                    flat[(d, h, s)] = j
    probe = {}
    for d in DS:
        for s in SEEDS:
            j = load(f'cube_probe_t_hr_d{d}_s{s}_n100.json')
            if j:
                probe[(d, s)] = j

    print('=' * 78)
    print(f'CUBE L1 flat grid  ({len(flat)} / 27 runs)')
    print('=' * 78)
    print(f'{"d":>4} {"h":>8} {"s42":>7} {"s43":>7} {"s44":>7} {"mean":>7}')
    argmax = {}
    for d in DS:
        cand = []
        for h in HORIZONS:
            vals = [flat[(d, h, s)]['success_rate']
                    for s in SEEDS if (d, h, s) in flat]
            if not vals:
                continue
            row = [f'{v:6.2f}' if (d, h, s) in flat else '     —'
                   for v in ([flat.get((d, h, s), {}).get('success_rate')
                              for s in SEEDS])]
            print(f'{d:>4} {str(h):>8} ' + ' '.join(row)
                  + f' {np.mean(vals):7.2f}')
            cand.append((np.mean(vals), h))
        if cand:
            argmax[d] = max(cand)

    print()
    print('=' * 78)
    print(f'CUBE L3 probe  ({len(probe)} / 9 runs)')
    print('=' * 78)
    for d in DS:
        vals = [probe[(d, s)]['success_rate']
                for s in SEEDS if (d, s) in probe]
        if vals:
            print(f'  d{d}: ' + ' '.join(f's{s}={probe[(d, s)]["success_rate"]:.2f}'
                                         for s in SEEDS if (d, s) in probe)
                  + f'  mean={np.mean(vals):.2f}')

    print()
    print('=' * 78)
    print('PAIRED McNEMAR  probe vs argmax(flat)  [per seed + pooled]')
    print('=' * 78)
    verdict = {}
    for d in DS:
        if d not in argmax:
            continue
        h_best = argmax[d][1]
        pooled_a, pooled_b = [], []
        for s in SEEDS:
            pa = probe.get((d, s))
            pb = flat.get((d, h_best, s))
            if not pa or not pb:
                continue
            a = np.asarray(pa['episode_successes'], bool)
            b = np.asarray(pb['episode_successes'], bool)
            r = mcnemar(a, b)
            if r is None:
                continue
            print(f'  d{d} s{s} vs flat{h_best}: a={a.mean():.3f} '
                  f'b={b.mean():.3f}  [{r[2]}:{r[3]}] p={r[1]:.4f}')
            pooled_a.append(a)
            pooled_b.append(b)
        if pooled_a:
            r = mcnemar(np.concatenate(pooled_a), np.concatenate(pooled_b))
            verdict[d] = {'h_best': h_best, 'probe': float(np.concatenate(pooled_a).mean()),
                          'flat': float(np.concatenate(pooled_b).mean()),
                          'p': r[1], 'n01': r[2], 'n10': r[3]}
            print(f'  d{d} POOLED vs flat{h_best}: probe={verdict[d]["probe"]:.3f} '
                  f'flat={verdict[d]["flat"]:.3f} [{r[2]}:{r[3]}] p={r[1]:.4f}')

    # ---- T' telemetry from logs ----
    print()
    print('=' * 78)
    print("PROBE T' TELEMETRY (parsed from logs)")
    print('=' * 78)
    tp_all = []
    for d in DS:
        for s in SEEDS:
            log = RUNS / f'cube_probe_t_hr_d{d}_s{s}_n100.log'
            if not log.exists():
                continue
            txt = log.read_text(errors='replace')
            Ts = [int(m) for m in re.findall(r"'T': (\d+)", txt)]
            fires = len(re.findall(r"'type': 'real_fire'", txt))
            replans = len(re.findall(r"'type': 'probe_t'", txt))
            if Ts:
                tp_all.extend(Ts)
                cnt = {t: Ts.count(t) for t in sorted(set(Ts))}
                print(f'  d{d} s{s}: replans={replans} fires={fires} '
                      f'T-dist={cnt} mean={np.mean(Ts):.2f}')
    if tp_all:
        cnt = {t: tp_all.count(t) for t in sorted(set(tp_all))}
        top = max(cnt.values()) / len(tp_all)
        print(f'\n  POOLED n={len(tp_all)} T-dist={cnt} '
              f'mean={np.mean(tp_all):.2f} dominant_share={top:.1%}')

    out = {
        'flat_runs': len(flat), 'probe_runs': len(probe),
        'argmax_per_d': {str(k): {'mean': v[0], 'h': v[1]}
                         for k, v in argmax.items()},
        'mcnemar_probe_vs_argmax': verdict,
        'tprime_pooled_n': len(tp_all),
        'tprime_dist': {str(t): tp_all.count(t) for t in sorted(set(tp_all))},
        'tprime_dominant_share': (max(tp_all.count(t) for t in set(tp_all))
                                  / len(tp_all)) if tp_all else None,
    }
    (RUNS / 'cube_verdict.json').write_text(json.dumps(out, indent=2))
    print(f'\n[written] {RUNS / "cube_verdict.json"}')


if __name__ == '__main__':
    main()
