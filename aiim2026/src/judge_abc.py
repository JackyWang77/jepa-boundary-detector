"""Judgement pass over the three 4060 lanes (A/B/C), 2026-11-06.

A: decisive triangle at d75 (claim b)   -> 3-seed means + McNemar (paired,
   same seed = same task set, cross-machine sampling verified identical)
B: sweet-spot grid d x h                -> 3-seed means, argmax h per d
C: cube E1 flat baseline                -> 3-seed means per d
plus: probe_t_hr replan telemetry (T' distribution, fires) for Fig 4.

Output: printed tables + runs/abc_verdict.json
"""
import json
import math
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / 'runs'
SEEDS = (42, 43, 44)


def load(name):
    with open(RUNS / name, encoding='utf-8') as f:
        return json.load(f)


def succ(d):
    return [bool(x) for x in d['episode_successes']]


def mcnemar(a, b):
    """Continuity-corrected McNemar on paired bool lists.
    Returns (n_a_only, n_b_only, p)."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2))   # survival of chi2_1
    return n01, n10, p


def mean3(vals):
    m = sum(vals) / len(vals)
    lo, hi = min(vals), max(vals)
    return m, lo, hi


def fmt(m, lo, hi):
    return f'{m:6.2f}  [{lo:.1f}..{hi:.1f}]'


verdict = {}

# ---------------- Lane A: triangle at d75 ----------------
print('=' * 72)
print('LANE A  decisive triangle, d=75, n=150/seed, seeds 42/43/44')
print('=' * 72)
arms_a = {
    'flat(5,5)': 'flat_h5r5_d75_s{s}_n150.json',
    'flat(10,10)': 'flat_h10r10_d75_s{s}_n150.json',
    'flat(15,15)': 'flat_h15r15_d75_s{s}_n150.json',
    'probe_t_hr': 'probe_t_hr_d75_s{s}_n150.json',
}
table_a = {}
succ_by_arm_seed = {}
for arm, pat in arms_a.items():
    per_seed, succs = [], {}
    for s in SEEDS:
        d = load(pat.format(s=s))
        per_seed.append(d['success_rate'])
        succs[s] = succ(d)
    table_a[arm] = {'per_seed': per_seed}
    succ_by_arm_seed[arm] = succs
    m, lo, hi = mean3(per_seed)
    table_a[arm]['mean'] = m
    table_a[arm]['range'] = [lo, hi]
    print(f'{arm:12s} {fmt(m, lo, hi)}   per-seed {[round(x,1) for x in per_seed]}')

print()
print('McNemar probe_t_hr vs flat (paired, same seed same tasks):')
mcn = {}
for flat in ('flat(5,5)', 'flat(10,10)', 'flat(15,15)'):
    merged_a, merged_b = [], []
    per_seed = {}
    for s in SEEDS:
        a, b = succ_by_arm_seed[flat][s], succ_by_arm_seed['probe_t_hr'][s]
        n01, n10, p = mcnemar(a, b)
        per_seed[s] = {'flat_only': n01, 'probe_only': n10, 'p': p}
        merged_a += a
        merged_b += b
    n01, n10, p = mcnemar(merged_a, merged_b)
    mcn[flat] = {'pooled': {'flat_only': n01, 'probe_only': n10, 'p': p},
                 'per_seed': per_seed}
    print(f'  vs {flat:11s} pooled n=450: flat-only={n01:3d} '
          f'probe-only={n10:3d}  p={p:.4f}')
verdict['A_triangle_d75'] = {'means': table_a, 'mcnemar_probe_vs': mcn}

# ---------------- Lane B: sweet-spot grid ----------------
print()
print('=' * 72)
print('LANE B  sweet-spot grid, n=150/seed (argmax h per d)')
print('=' * 72)
grid_pat = 'flat_h{h}r{h}_d{d}_s{s}_n150.json'
grid = {}
hdr = '        ' + ''.join(f'  h={h:<12d}' for h in (5, 10, 15))
print('         ' + ''.join(f'  h={h:<14d}' for h in (5, 10, 15)))
for d in (25, 50, 75):
    row = {}
    cells = []
    for h in (5, 10, 15):
        pat = grid_pat.format(h=h, d=d, s=42)
        if not (RUNS / pat).exists():
            cells.append(f'{"—":>16s}')
            continue
        per_seed = [load(grid_pat.format(h=h, d=d, s=s))['success_rate']
                    for s in SEEDS]
        m, lo, hi = mean3(per_seed)
        row[h] = {'mean': m, 'per_seed': per_seed}
        cells.append(f'{m:6.2f}[{lo:.0f}-{hi:.0f}]  ')
    grid[d] = row
    best_h = max(row, key=lambda h: row[h]['mean'])
    print(f'd={d:<3d}   ' + ''.join(cells) + f' argmax h={best_h}')
verdict['B_grid'] = {str(d): {str(h): v for h, v in row.items()}
                     for d, row in grid.items()}

# h5-vs-h10 gap per d (does the horizon cost shrink with d?)
print()
for d in (25, 50, 75):
    if 5 in grid[d] and 10 in grid[d]:
        gap = grid[d][5]['mean'] - grid[d][10]['mean']
        print(f'  d={d:<3d}  h5 - h10 = {gap:+.2f} pts')

# ---------------- Lane C: cube E1 flat baseline ----------------
print()
print('=' * 72)
print('LANE C  cube (OGB) E1 flat baseline, n=50/seed')
print('=' * 72)
cube = {}
for d in (25, 50, 75):
    per_seed = [load(f'cube_flat_d{d}_s{s}.json')['success_rate']
                for s in SEEDS]
    m, lo, hi = mean3(per_seed)
    cube[d] = {'mean': m, 'range': [lo, hi], 'per_seed': per_seed}
    print(f'd={d:<3d}  {fmt(m, lo, hi)}   per-seed {[round(x,1) for x in per_seed]}')
verdict['C_cube_flat'] = {str(d): v for d, v in cube.items()}

# ---------------- probe telemetry (Fig 4) ----------------
print()
print('=' * 72)
print('PROBE TELEMETRY  probe_t_hr d75 (keys of one run first)')
print('=' * 72)
d0 = load('probe_t_hr_d75_s42_n150.json')
print('top-level keys:', sorted(d0.keys()))
tele = {}
for key in d0:
    v = d0[key]
    if isinstance(v, list) and v and isinstance(v[0], (int, float)) \
            and key not in ('episode_successes',):
        arr = v
        tele[key] = {
            'n': len(arr),
            'mean': sum(arr) / len(arr),
            'min': min(arr),
            'max': max(arr),
        }
        print(f'  {key:24s} n={len(arr):5d} mean={tele[key]["mean"]:.2f} '
              f'min={tele[key]["min"]} max={tele[key]["max"]}')
verdict['probe_telemetry_keys'] = sorted(d0.keys())
verdict['probe_telemetry_scalar_lists'] = tele

with open(RUNS / 'abc_verdict.json', 'w', encoding='utf-8') as f:
    json.dump(verdict, f, indent=2, ensure_ascii=False)
print()
print('saved -> runs/abc_verdict.json')
