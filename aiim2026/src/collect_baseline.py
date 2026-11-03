"""Aggregate the 9 flat-planning baseline logs and compare with published
Mind the Gap Table 3 numbers (LeWM on PushT: 94.0 / 52.7 / 18.0).

Usage: python collect_baseline.py  ->  runs/baseline_summary.json + verdict
"""

import json
import re
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[1] / 'runs'
PUBLISHED = {25: 94.0, 50: 52.7, 75: 18.0}
PUBLISHED_STD = {25: 2.0, 50: 5.0, 75: 2.0}
SEEDS = (42, 43, 44)


def parse_log(path):
    text = path.read_text()
    m = re.search(r"'success_rate': ([\d.]+)", text)
    return float(m.group(1)) if m else None


def main():
    table = {}
    for d in (25, 50, 75):
        rates = []
        for seed in SEEDS:
            p = RUNS / f'flat_d{d}_seed{seed}.log'
            rate = parse_log(p) if p.exists() else None
            if rate is None:
                print(f'MISSING: {p.name}')
            rates.append(rate)
        found = [r for r in rates if r is not None]
        if found:
            table[d] = {
                'per_seed': rates,
                'mean': float(np.mean(found)),
                'std': float(np.std(found)),
                'n': len(found),
                'published': PUBLISHED[d],
                'published_std': PUBLISHED_STD[d],
            }

    print(f'{"d":>4} {"ours (mean+-std)":>20} {"published":>12} {"delta":>7}')
    for d, row in table.items():
        print(f'{d:>4} {row["mean"]:>10.1f} +- {row["std"]:<6.1f} '
              f'{row["published"]:>8.1f} +- {row["published_std"]:<3.1f} '
              f'{row["mean"] - row["published"]:>7.1f}')

    # verdict: within 1.5 published std on d=75 and d=50, monotone collapse
    checks = {}
    for d in (25, 50, 75):
        if d in table:
            gap = abs(table[d]['mean'] - PUBLISHED[d])
            checks[f'd{d}_within_1.5std'] = bool(
                gap <= 1.5 * PUBLISHED_STD[d])
    means = [table[d]['mean'] for d in (25, 50, 75) if d in table]
    checks['monotone_collapse'] = bool(
        means[0] > means[1] > means[2]) if len(means) == 3 else False
    checks['all_9_logs'] = all(
        (RUNS / f'flat_d{d}_seed{s}.log').exists()
        for d in (25, 50, 75) for s in SEEDS)
    verdict = all(checks.values())
    print('\nchecks:', json.dumps(checks, indent=2))
    print('VERDICT:', 'REPRODUCED' if verdict else 'CHECK GAPS')

    out = {'table': table, 'checks': checks,
           'verdict': 'REPRODUCED' if verdict else 'CHECK GAPS'}
    with open(RUNS / 'baseline_summary.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('-> runs/baseline_summary.json')


if __name__ == '__main__':
    main()
