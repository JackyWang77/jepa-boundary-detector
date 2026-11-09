"""PACE baseline runner: 18 runs (d x seed x theta) on the PushT protocol.

Protocol matches the flat grid and the probe arm exactly:
  eval_budget = 2 * goal_offset_steps, num_eval=150, history_len=3,
  action_block=5, solver on cuda, arm=pace.

PACE is training-free: no detector, no latent probe. Each CEM query returns
the full 10-block chunk and the executed prefix is chosen from the speed
profile (see pace_arm.py). Two theta values are run as specified in
EXPERIMENTS_4060.md.

Resume-safe (skips existing jsons), aborts on first failure.
Usage:
  python src/run_pace.py            # both theta values
  python src/run_pace.py --theta 0.5
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
DS = (25, 50, 75)
THETAS = (0.5, 0.3)
N_EVAL = 150
MIN_BLOCKS = 2


def tag(theta):
    return f'{theta:.2f}'.replace('.', '')


def run_one(d, seed, theta):
    name = f'pace_th{tag(theta)}_d{d}_s{seed}_n{N_EVAL}'
    out_json = RUNS / f'{name}.json'
    if out_json.exists():
        print(f'[skip] {name}', flush=True)
        return True
    budget = 2 * d
    cmd = [PY, '-u', str(ROOT / 'src' / 'e6_eval.py'),
           '--config-name=pusht.yaml', 'policy=pusht/weights.pt',
           f'eval.goal_offset_steps={d}',
           f'eval.eval_budget={budget}', f'eval.num_eval={N_EVAL}',
           '+plan_config.history_len=3',
           'plan_config.horizon=10', 'plan_config.receding_horizon=10',
           'plan_config.action_block=5', 'solver.device=cuda',
           f'seed={seed}',
           f'+out_json={out_json}', '+arm=pace',
           f'+pace_theta={theta}', f'+pace_min_blocks={MIN_BLOCKS}']
    print(f'[run ] {name} budget={budget}', flush=True)
    log = RUNS / f'{name}.log'
    t0 = time.time()
    with open(log, 'w') as f:
        rc = subprocess.run(cmd, cwd=PLAN, stdout=f,
                            stderr=subprocess.STDOUT).returncode
    dt = (time.time() - t0) / 60
    if rc != 0 or not out_json.exists():
        print(f'[FAIL] {name} rc={rc} after {dt:.1f}min -> abort chain',
              flush=True)
        return False
    sr = json.loads(out_json.read_text())['success_rate']
    print(f'[done] {name} {sr:.1f}% ({dt:.1f}min)', flush=True)
    return True


def main():
    chosen = list(THETAS)
    if '--theta' in sys.argv:
        chosen = [float(sys.argv[sys.argv.index('--theta') + 1])]
    os.environ.setdefault('STABLEWM_HOME', r'D:\stable_worldmodel')
    RUNS.mkdir(exist_ok=True)
    t0 = time.time()
    done = fail = 0
    for theta in chosen:
        for d in DS:
            for seed in SEEDS:
                if run_one(d, seed, theta):
                    done += 1
                else:
                    fail += 1
                    print(f'[SUMMARY] done={done} failed={fail} -> abort',
                          flush=True)
                    sys.exit(1)
    print(f'[SUMMARY] all {done} runs done in '
          f'{(time.time() - t0) / 3600:.2f}h', flush=True)


if __name__ == '__main__':
    main()
