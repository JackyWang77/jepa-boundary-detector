"""Generate per-seed success-rate tables for PushT and cube.

Reads all flat + probe result jsons in runs/ and emits one table per task:
  rows = arm (h5r5/h10r10/h15r15/probe), columns = d x seed
Meant to back the paper table footnote ("std across seeds").

Usage: python make_per_seed_table.py
Writes results/per_seed_pusht.csv and results/per_seed_cube.csv and prints md.
"""
import glob, json, re, statistics as st
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'runs'

SEEDS = (42, 43, 44)


def _read(path):
    try:
        return json.load(open(path, encoding='utf-8'))
    except (OSError, ValueError):
        return None


def collect(task, pattern, arm_parse):
    """Return {(d, seed): (success_rate, n_ep)} for a task+pattern.
    arm_parse(name) -> arm label."""
    out = {}
    for p in glob.glob(str(RUNS / pattern)):
        j = _read(p)
        if not j:
            continue
        name = Path(p).stem
        d = j.get('goal_offset_steps')
        seed = j.get('seed')
        succ = j.get('success_rate')
        n = len(j.get('episode_successes', []))
        if d is None or seed is None or succ is None:
            continue
        arm = arm_parse(name)
        out[(arm, d, seed)] = (succ, n)
    return out


def arm_flat(name):
    m = re.search(r'flat_h(\d+)r(\d+)', name)
    return f'h{m.group(1)}r{m.group(2)}' if m else name


def arm_probe(name):
    return 'probe'


def build_table(data, label):
    """data: {(arm,d,seed): (sr,n)}. Print md table and return csv rows."""
    arms = sorted({k[0] for k in data})
    ds = sorted({k[1] for k in data})
    rows = []
    header = ['arm'] + [f'd{d}_s{s}' for d in ds for s in SEEDS] + \
             [f'd{d}_mean' for d in ds] + [f'd{d}_std' for d in ds]
    print(f'\n===== {label} =====')
    print('| arm | ' + ' | '.join(f'd{d}_s{s}' for d in ds for s in SEEDS) +
          ' | ' + ' | '.join(f'd{d}_mean' for d in ds) +
          ' | ' + ' | '.join(f'd{d}_std' for d in ds) + ' |')
    print('|' + '---|' * (1 + len(ds)*len(SEEDS) + 2*len(ds)))
    for arm in arms:
        row = [arm]
        means, stds = [], []
        for d in ds:
            vals = []
            for s in SEEDS:
                v = data.get((arm, d, s))
                row.append(f'{v[0]:.1f}' if v else '—')
                if v:
                    vals.append(v[0])
            means.append(st.mean(vals) if vals else float('nan'))
            stds.append(st.pstdev(vals) if len(vals) >= 2 else 0.0)
        row += [f'{m:.1f}' for m in means] + [f'{s:.1f}' for s in stds]
        rows.append(row)
        print('| ' + ' | '.join(row) + ' |')
    return header, rows


def main():
    # PushT
    flat = collect('pusht', 'flat_h*r*_d*_s*_n150.json', arm_flat)
    probe = collect('pusht', 'probe_t_hr_d*_s*_n150.json', arm_probe)
    pusht = {**flat, **probe}
    h, r = build_table(pusht, 'PushT (n=150/seed)')
    with open(ROOT / 'results' / 'per_seed_pusht.csv', 'w', encoding='utf-8') as f:
        f.write(','.join(h) + '\n')
        for row in r:
            f.write(','.join(row) + '\n')

    # cube
    cflat = collect('cube', 'cube_flat_h*r*_d*_s*_n100.json', arm_flat)
    cprobe = collect('cube', 'cube_probe_t_hr_d*_s*_n100.json', arm_probe)
    cube = {**cflat, **cprobe}
    h2, r2 = build_table(cube, 'cube (n=100/seed)')
    with open(ROOT / 'results' / 'per_seed_cube.csv', 'w', encoding='utf-8') as f:
        f.write(','.join(h2) + '\n')
        for row in r2:
            f.write(','.join(row) + '\n')

    # seed-mean/std summary for footnote
    print('\n----- per-seed std summary (mean of within-d std across arms) -----')
    for tag, data in [('PushT', pusht), ('cube', cube)]:
        stds = []
        for arm in sorted({k[0] for k in data}):
            for d in sorted({k[1] for k in data}):
                vals = [data[(arm, d, s)][0] for s in SEEDS if (arm, d, s) in data]
                if len(vals) >= 2:
                    stds.append(st.pstdev(vals))
        if stds:
            print(f'  {tag}: avg per-(arm,d) seed-std = {st.mean(stds):.2f}pp, '
                  f'max = {max(stds):.2f}pp')


if __name__ == '__main__':
    main()
