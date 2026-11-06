"""Final cross-lane Mcnemar for PushT: probe (A+D) vs flat grid (B)."""
import json, math, sys
from pathlib import Path
import numpy as np

RUNS = Path(r'd:\Research-2026-jepa-boundary-detector\aiim2026\runs')

# Load lane B flat raw episode_successes
B = {}  # (seed, d, arm) -> bool array
for h, arm in [('h5r5', 'flat(5,5)'), ('h10r10', 'flat(10,10)'), ('h15r15', 'flat(15,15)')]:
    for d in ['25', '50', '75']:
        for seed in [42, 43, 44]:
            matches = list(RUNS.glob(f'*{h}_d{d}_s{seed}_n150.json'))
            if not matches: continue
            j = json.loads(matches[0].read_text())
            B[(seed, d, arm)] = np.array(j['episode_successes'], dtype=bool)

# Load probe d25/d50/d75 for all seeds
PROBE = {}  # (seed, d) -> bool array
for d in ['25', '50', '75']:
    for seed in [42, 43, 44]:
        matches = list(RUNS.glob(f'probe_t_hr_d{d}_s{seed}_n150.json'))
        if not matches: continue
        j = json.loads(matches[0].read_text())
        PROBE[(seed, d)] = np.array(j['episode_successes'], dtype=bool)

def mcnemar(a, b):
    tab = np.zeros((2,2), dtype=int)
    for ai, bi in zip(a, b):
        tab[int(ai), int(bi)] += 1
    bonly = tab[0,1]; aonly = tab[1,0]
    if bonly + aonly < 25: return None
    chi2 = (abs(bonly - aonly) - 1)**2 / (bonly + aonly) if (bonly+aonly) else 0
    p = math.erfc(math.sqrt(chi2/2))
    return {'probe_only':int(aonly), 'flat_only':int(bonly), 'chi2':chi2, 'p':p}

print("="*70)
print("PushT ALL LANES · SUCCESS RATES per (d, seed)")
print("="*70)
for d in ['25', '50', '75']:
    print(f"\n--- d={d} ---")
    for seed in [42, 43, 44]:
        parts = [f"s{seed}"]
        for arm in ['flat(5,5)', 'flat(10,10)', 'flat(15,15)']:
            k = (seed, d, arm)
            if k in B:
                parts.append(f"{arm}={B[k].mean()*100:.1f}%")
        pk = (seed, d)
        if pk in PROBE:
            parts.append(f"probe={PROBE[pk].mean()*100:.1f}%")
        print("  " + "  ".join(parts))
    # means
    means = {}
    for arm in ['flat(5,5)', 'flat(10,10)', 'flat(15,15)']:
        vals = [B[(s,d,arm)].mean()*100 for s in [42,43,44] if (s,d,arm) in B]
        if vals: means[arm] = (np.mean(vals), np.min(vals), np.max(vals))
    pvals = [PROBE[(s,d)].mean()*100 for s in [42,43,44] if (s,d) in PROBE]
    means['probe_t_hr'] = (np.mean(pvals), np.min(pvals), np.max(pvals))
    print(f"  MEANS: " + "  ".join(f"{k}={v[0]:.1f}[{v[1]:.1f}-{v[2]:.1f}]" for k,v in means.items()))

print("\n" + "="*70)
print("PAIRED McNEMAR — probe_t_hr vs flat (per seed + pooled n=450)")
print("="*70)
for d in ['25', '50', '75']:
    print(f"\n--- d={d} ---")
    pool = {'flat(5,5)': np.array([], dtype=bool), 'flat(10,10)': np.array([], dtype=bool), 'probe': np.array([], dtype=bool)}
    for seed in [42, 43, 44]:
        ps = PROBE.get((seed,d)); f5 = B.get((seed,d,'flat(5,5)')); f10 = B.get((seed,d,'flat(10,10)'))
        if ps is None: continue
        line = f"  s{seed}: probe={ps.mean()*100:.1f}%"
        pool['probe'] = np.concatenate([pool['probe'], ps]) if pool['probe'].size else ps
        if f5 is not None:
            m5 = mcnemar(ps, f5)
            pool['flat(5,5)'] = np.concatenate([pool['flat(5,5)'], f5]) if pool['flat(5,5)'].size else f5
            line += f"  vs h5={f5.mean()*100:.1f}%" + (f" [{m5['probe_only']}v{m5['flat_only']} p={m5['p']:.3f}]" if m5 else " (n<25)")
        if f10 is not None:
            m10 = mcnemar(ps, f10)
            pool['flat(10,10)'] = np.concatenate([pool['flat(10,10)'], f10]) if pool['flat(10,10)'].size else f10
            line += f"  vs h10={f10.mean()*100:.1f}%" + (f" [{m10['probe_only']}v{m10['flat_only']} p={m10['p']:.3f}]" if m10 else " (n<25)")
        print(line)
    # pooled
    ps = pool['probe']
    for arm in ['flat(5,5)', 'flat(10,10)', 'flat(15,15)']:
        f = pool.get(arm, np.array([], dtype=bool))
        if f.size:
            m = mcnemar(ps, f)
            sig = '***' if m['p']<1e-3 else '**' if m['p']<0.01 else '*' if m['p']<0.05 else 'ns'
            print(f"  POOLED probe={ps.mean()*100:.1f}% vs {arm}={f.mean()*100:.1f}%  [{m['probe_only']}v{m['flat_only']}] p={m['p']:.6f} {sig}")

print("\n" + "="*70)
print("HORIZON COST Δ = h5 - h10 (pp) — bigger = wrong horizon hurts more")
print("="*70)
grid = json.loads(Path(r'd:\Research-2026-jepa-boundary-detector\aiim2026\runs\abc_verdict.json').read_text())['B_grid']
for d in ['25','50','75']:
    g5 = grid[d]['5']['mean']; g10 = grid[d]['10']['mean']
    g15 = grid[d].get('15', {}).get('mean', None)
    line = f"  d{d}: h5={g5:.1f}  h10={g10:.1f}  Δ(h5-h10)={g5-g10:+.1f}pp"
    if g15 is not None: line += f"  h15={g15:.1f}"
    print(line)

# Wall-clock: load from task D data
print("\n" + "="*70)
print("WALL-CLOCK · probe vs flat (lane A)")
print("="*70)
A = json.loads(Path(r'd:/Research-2026-jepa-boundary-detector/taskA_data.js').read_text().replace('window.TASKA = ', ''))
print(f"  flat(5,5):  s42=615m s43=394m s44=196m  total={sum(r['elapsed_min'] for r in A['runs'] if 'flat_h5r5' in r['name']):.0f}m")
print(f"  flat(10,10): s42=580m s43=347m s44=153m  total={sum(r['elapsed_min'] for r in A['runs'] if 'flat_h10r10' in r['name']):.0f}m")
print(f"  probe_t_hr: s42=522m s43=301m s44=103m  total={sum(r['elapsed_min'] for r in A['runs'] if 'probe' in r['name']):.0f}m")

# T' telemetry
print("\n" + "="*70)
print("T' TELEMETRY (lane A probe_t_hr · n=3487 replans)")
print("="*70)
tps = json.loads(Path(r'd:\Research-2026-jepa-boundary-detector\aiim2026\runs\probe_tprime_stats.json').read_text())
h = tps['pooled_T_hist']; total = sum(h.values())
print(f"  T=5: {h.get(5,0)} ({100*h.get(5,0)/total:.1f}%)")
print(f"  T=10: {h.get(10,0)} ({100*h.get(10,0)/total:.1f}%)")
print(f"  T=11: {h.get(11,0)}")
print(f"  mean T' = {tps['mean_T_prime']}")
print(f"  fires={tps['pooled_fires']} replans={tps['pooled_replans']}")

print("\n" + "="*70)
print("CUBE LANE C · SUCCESS RATES")
print("="*70)
C = json.loads(Path(r'd:/Research-2026-jepa-boundary-detector/aiim2026\runs\abc_verdict.json').read_text())['C_cube_flat']
for d in ['25','50','75']:
    c = C[d]
    print(f"  d{d}: mean={c['mean']:.1f}%  [{c['range'][0]:.0f}-{c['range'][1]:.0f}]  (local h5 polluted)")
