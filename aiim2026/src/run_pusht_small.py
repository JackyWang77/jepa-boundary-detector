"""R2 cross-capacity runner: flat grid + zero-shot probe on the halved LeWM.

The small model lives at $STABLEWM_HOME/checkpoints/pusht_small/{config.json,
weights.pt}. Everything else (protocol, budgets, seeds, detector head, fz,
thresholds) is identical to the main-model PushT grid, so the only variable is
predictor capacity (depth 6->3, mlp_dim 2048->1024).

  python src/run_pusht_small.py --h 10      # flat h=10, d in {50,75}, 3 seeds
  python src/run_pusht_small.py --probe     # zero-shot probe, d in {50,75}, 3 seeds
  python src/run_pusht_small.py --chain --h 10
      # idle until this lane's 6 flat jsons exist, then run its share of the
      # probe grid (one seed per lane) so no GPU time is lost between phases

Resume-safe (skips existing jsons), aborts on first failure.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # aiim2026/
PLAN = ROOT / 'third_party' / 'stable-worldmodel' / 'scripts' / 'plan'
RUNS = ROOT / 'runs'
PY = os.environ.get('AIIM_PY', r'C:\Users\Ghazz\miniconda3\envs\aiim\python.exe')

SEEDS = (42, 43, 44)
DS = (50, 75)
N_EVAL = 150
MODEL = 'pusht_small/weights.pt'
H10 = ['plan_config.horizon=10', 'plan_config.receding_horizon=10']
# probe runs handed to each flat lane, so the probe grid starts as soon as that
# lane is done instead of waiting for all 18 flat runs
CHAIN_SEEDS = {5: (42,), 10: (43,), 15: (44,)}


def wait_for_flat(h, timeout_h=14):
    want = [RUNS / f'pusht_small_flat_h{h}_d{d}_s{s}.json'
            for d in DS for s in SEEDS]
    t0 = time.time()
    while True:
        missing = [p.name for p in want if not p.exists()]
        if not missing:
            print(f'[chain] h={h} flat grid complete', flush=True)
            return True
        if (time.time() - t0) / 3600 > timeout_h:
            print(f'[chain] timeout, missing {missing}', flush=True)
            return False
        time.sleep(60)


def run_one(name, d, seed, extra):
    out_json = RUNS / f'{name}.json'
    if out_json.exists():
        print(f'[skip] {name}', flush=True)
        return True
    cmd = [PY, '-u', str(ROOT / 'src' / 'e6_eval.py'),
           '--config-name=pusht.yaml', f'policy={MODEL}',
           f'eval.goal_offset_steps={d}', f'eval.eval_budget={2 * d}',
           f'eval.num_eval={N_EVAL}',
           '+plan_config.history_len=3',
           'plan_config.action_block=5', 'solver.device=cuda',
           f'seed={seed}', f'+out_json={out_json}'] + extra
    print(f'[run ] {name}', flush=True)
    t0 = time.time()
    with open(RUNS / f'{name}.log', 'w') as f:
        rc = subprocess.run(cmd, cwd=PLAN, stdout=f,
                            stderr=subprocess.STDOUT).returncode
    dt = (time.time() - t0) / 60
    if rc != 0 or not out_json.exists():
        print(f'[FAIL] {name} rc={rc} after {dt:.1f}min -> abort', flush=True)
        return False
    sr = json.loads(out_json.read_text())['success_rate']
    print(f'[done] {name} {sr:.1f}% ({dt:.1f}min)', flush=True)
    return True


def main():
    os.environ.setdefault('STABLEWM_HOME', r'D:\stable_worldmodel')
    RUNS.mkdir(exist_ok=True)
    probe = '--probe' in sys.argv
    probe2 = '--probe2' in sys.argv
    chain = '--chain' in sys.argv
    if probe:
        tasks = [(f'pusht_small_probe_d{d}_s{s}', d, s,
                  H10 + ['+arm=probe_t'])
                 for d in DS for s in SEEDS]
    elif probe2:
        # Phase 3: re-distilled head trained on small-model features
        tasks = [(f'pusht_small_probe2_d{d}_s{s}', d, s,
                  H10 + ['+arm=probe_t',
                         '+detector_meta=r2_dual_small_meta.json'])
                 for d in DS for s in SEEDS]
    elif chain:
        hs = ([int(sys.argv[sys.argv.index('--h') + 1])] if '--h' in sys.argv
              else sorted(CHAIN_SEEDS))
        tasks = []
        for h in hs:
            if not wait_for_flat(h):
                sys.exit(1)
            tasks += [(f'pusht_small_probe_d{d}_s{s}', d, s,
                       H10 + ['+arm=probe_t'])
                      for d in DS for s in CHAIN_SEEDS[h]]
    else:
        h = int(sys.argv[sys.argv.index('--h') + 1])
        tasks = [(f'pusht_small_flat_h{h}_d{d}_s{s}', d, s,
                  [f'plan_config.horizon={h}',
                   f'plan_config.receding_horizon={h}'])
                 for d in DS for s in SEEDS]
    t0 = time.time()
    for name, d, seed, extra in tasks:
        if not run_one(name, d, seed, extra):
            sys.exit(1)
    print(f'[SUMMARY] {len(tasks)} runs in {(time.time() - t0) / 3600:.2f}h',
          flush=True)


if __name__ == '__main__':
    main()
