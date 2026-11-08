"""PushT probe_t v2 rerun: 9 runs (d25/d50/d75 x seeds 42/43/44), n=150.

Why v2 (supersedes archive_probe_v1_buggy/):
  1. probe_t_policy.py had two bugs (fixed 2026-11-07):
     - real-stream refractory compared in ENV STEPS instead of detector
       rows -> effective refractory 1 row instead of 5 (over-firing:
       45% of adjacent fire gaps were 1 block on d75)
     - T' had no upper bound; T'=11 overflowed the action deque
       (maxlen=PROBE_H*block) silently dropping the first block
  2. run_taskD.py ran d25/d50 with eval_budget=150 instead of 2*d
     (50/100) -> not comparable with the flat grid whose budget is 2d.
     d75 was the only correctly-budgeted probe arm in v1.

Protocol (matches flat grid + cube probe):
  eval_budget = 2 * goal_offset_steps, num_eval=150, horizon=receding=10,
  history_len=3, action_block=5, solver on cuda, arm=probe_t.

Waits for the cube probe grid (9 jsons) to finish first so the two
experiments never share the 8GB GPU. Resume-safe (skips existing jsons),
aborts on first failure.
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
PY = r'C:\Users\Ghazz\miniconda3\envs\aiim\python.exe'

SEEDS = (42, 43, 44)
DS = (25, 50, 75)
N_EVAL = 150
CUBE_PROBE_GLOB = 'cube_probe_t_hr_*_n100.json'
CUBE_PROBE_TOTAL = 9
WAIT_S = 60
WAIT_MAX_H = 10


def wait_for_cube(glob_pat=CUBE_PROBE_GLOB):
    t0 = time.time()
    while True:
        done = len(list(RUNS.glob(glob_pat)))
        if done >= CUBE_PROBE_TOTAL:
            print(f'[gate] cube probe {done}/{CUBE_PROBE_TOTAL} -> start PushT',
                  flush=True)
            return
        if time.time() - t0 > WAIT_MAX_H * 3600:
            print('[gate] timeout waiting for cube probe; starting anyway',
                  flush=True)
            return
        print(f'[gate] cube probe {done}/{CUBE_PROBE_TOTAL}, waiting...',
              flush=True)
        time.sleep(WAIT_S)


def run_one(d, seed, t1=False):
    tag = 't1_hr' if t1 else 't_hr'
    name = f'probe_{tag}_d{d}_s{seed}_n{N_EVAL}'
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
           f'+out_json={out_json}', '+arm=probe_t']
    if t1:
        cmd.append('+t_plus_one=true')
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
    t1 = '--t1' in sys.argv
    os.environ.setdefault('STABLEWM_HOME', r'D:\stable_worldmodel')
    RUNS.mkdir(exist_ok=True)
    gate = 'cube_probe_t1_hr_*_n100.json' if t1 else CUBE_PROBE_GLOB
    wait_for_cube(gate)
    t0 = time.time()
    done = fail = 0
    for d in DS:
        for seed in SEEDS:
            ok = run_one(d, seed, t1=t1)
            if ok:
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
