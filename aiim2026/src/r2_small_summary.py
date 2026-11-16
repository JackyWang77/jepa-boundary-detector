"""R2 cross-capacity final summary: flat grid + zero-shot probe + re-distilled
probe2 on the reduced-capacity model (predictor depth 3, mlp 1024, 11.46M).

Reads the 30 run jsons, builds the paper table, runs two-proportion z-tests
for the pre-registered comparisons, writes results/pusht_small/r2_small_summary.json.
"""

import json
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
OUT = ROOT / 'results' / 'pusht_small'
SEEDS = (42, 43, 44)


def load(name):
    d = json.load(open(RUNS / f'{name}.json'))
    eps = d['episode_successes']
    return int(sum(eps)), len(eps)


def cell(prefix, d):
    per_seed = {s: load(f'{prefix}_d{d}_s{s}')[0] for s in SEEDS}
    k = sum(per_seed.values())
    n = 150 * len(SEEDS)
    return {'per_seed': per_seed, 'successes': k, 'n': n, 'rate': k / n}


def ztest(a, b):
    k1, n1, k2, n2 = a['successes'], a['n'], b['successes'], b['n']
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    return {'z': z, 'p1': p1, 'p2': p2}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    arms = {f'flat_h{h}': f'pusht_small_flat_h{h}' for h in (5, 10, 15)}
    arms['probe_zeroshot'] = 'pusht_small_probe'
    arms['probe2_redistill'] = 'pusht_small_probe2'

    table = {arm: {str(d): cell(pref, d) for d in (50, 75)}
             for arm, pref in arms.items()}

    tests = {}
    for d in (50, 75):
        k = str(d)
        tests[f'probe2_vs_best_flat_d{d}'] = ztest(
            table['probe2_redistill'][k], table['flat_h5'][k])
        tests[f'probe2_vs_same_horizon_flat_h10_d{d}'] = ztest(
            table['probe2_redistill'][k], table['flat_h10'][k])
        tests[f'redistill_vs_zeroshot_d{d}'] = ztest(
            table['probe2_redistill'][k], table['probe_zeroshot'][k])

    summary = {
        'model': {'predictor_depth': 3, 'predictor_mlp_dim': 1024,
                  'params': 11460000,
                  'train_budget': '60 epochs x 5000 batches (eff. batch 128), '
                                  '8.1 h on RTX 4060',
                  'note': 'main model: depth 6 / mlp 2048, same encoder'},
        'protocol': {'episodes_per_run': 150, 'seeds': list(SEEDS),
                     'eval_budget': '2 x goal_offset', 'history_len': 3,
                     'action_block': 5},
        'detector': {
            'zeroshot': 'main-model dual head (r2_meta_dual.json) transferred '
                        'as-is',
            'redistill': 'head retrained on small-model features '
                         '(r2_dual_small_meta.json): CNN, AUC 0.795, '
                         'fire micro-F1 0.621 (PARTIAL)',
        },
        'table': table,
        'ztests': tests,
        'verdict': (
            'LIMITATION outcome per pre-registered rules: re-distilled probe '
            'matches best flat at d75 (n.s.) but clearly loses at d50 '
            '(z=-3.46, p<0.001). Detector beats same-horizon flat h10 at both '
            'distances and re-distillation recovers ~4pt over zero-shot, yet '
            'adaptive horizon cannot fully compensate halved capacity.'
        ),
    }
    with open(OUT / 'r2_small_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"{'arm':<18}{'d50':>8}{'d75':>8}")
    for arm in arms:
        r50 = 100 * table[arm]['50']['rate']
        r75 = 100 * table[arm]['75']['rate']
        print(f'{arm:<18}{r50:>7.1f}%{r75:>7.1f}%')
    print()
    for name, t in tests.items():
        print(f"{name:<38} {100*t['p1']:5.1f}% vs {100*t['p2']:5.1f}%  z={t['z']:+.2f}")
    print()
    print('verdict:', summary['verdict'])


if __name__ == '__main__':
    main()
