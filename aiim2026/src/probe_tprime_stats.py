"""T' distribution + re-anchor cadence from lane A probe replan_logs.

Feeds the compute-cost angle: a shorter chosen horizon means a cheaper
CEM solve per re-anchor. Reports the T' histogram and, for context, the
per-episode fire cadence.
"""
import json
from collections import Counter
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / 'runs'

hist = Counter()
fires_total = replans_total = 0
per_seed = {}
for s in (42, 43, 44):
    d = json.load(open(RUNS / f'probe_t_hr_d75_s{s}_n150.json'))
    seed_hist = Counter()
    f = r = 0
    for eps in d['replan_logs'].values():
        for seg in eps:                     # list of segments per episode
            for rep in seg['replans']:
                seed_hist[rep['T']] += 1
                r += 1
            f += len(seg.get('fires', []))
    hist += seed_hist
    fires_total += f
    replans_total += r
    per_seed[s] = {'hist': dict(seed_hist), 'fires': f, 'replans': r}

total = sum(hist.values())
mean_T = sum(t * n for t, n in hist.items()) / total
out = {
    'per_seed': per_seed,
    'pooled_T_hist': {str(t): n for t, n in sorted(hist.items())},
    'pooled_replans': replans_total,
    'pooled_fires': fires_total,
    'mean_T_prime': round(mean_T, 2),
    'share_T5_or_less': round(sum(n for t, n in hist.items() if t <= 5)
                              / total, 3),
}
print(json.dumps(out, indent=2))
with open(RUNS / 'probe_tprime_stats.json', 'w') as f:
    json.dump(out, f, indent=2)
