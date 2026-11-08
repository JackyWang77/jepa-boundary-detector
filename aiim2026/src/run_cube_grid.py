"""Cube grid runner: flat L1 (27 runs) and probe L3 (9 runs), parallel-capable.

Usage:
  python run_cube_grid.py --phase flat  --workers 2
  python run_cube_grid.py --phase probe --workers 2
  python run_cube_grid.py --phase flat  --workers 2 --smoke   # n=8, h5 only
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault('STABLEWM_HOME', r'D:\stable_worldmodel')
os.environ.setdefault('MUJOCO_GL', 'glfw')

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'third_party' / 'stable-worldmodel' / 'scripts' / 'plan'
PY = sys.executable
RUNS = ROOT / 'runs'
DATASET = 'D:/stable_worldmodel/datasets/cube_single_expert.h5'
H5S = ['5,5', '10,10', '15,15']
DS = [25, 50, 75]
SEEDS = [42, 43, 44]


def build_jobs(phase, n_eval, t1=False):
    jobs = []
    if phase == 'flat':
        for d in DS:
            for h in H5S:
                hh, rr = (int(x) for x in h.split(','))
                for s in SEEDS:
                    name = f'cube_flat_h{hh}r{rr}_d{d}_s{s}_n{n_eval}'
                    jobs.append(dict(
                        name=name, n=n_eval, seed=s,
                        overrides=[f'eval.goal_offset_steps={d}',
                                   f'eval.eval_budget={2 * d}',
                                   f'plan_config.horizon={hh}',
                                   f'plan_config.receding_horizon={rr}']))
    else:
        for d in DS:
            for s in SEEDS:
                tag = 't1_hr' if t1 else 't_hr'
                name = f'cube_probe_{tag}_d{d}_s{s}_n{n_eval}'
                ov = [f'eval.goal_offset_steps={d}',
                      f'eval.eval_budget={2 * d}',
                      'plan_config.horizon=10',
                      'plan_config.receding_horizon=10',
                      '+arm=probe_t',
                      '+features_dir=features_cube',
                      '+detector_meta=cube_r2_meta_dual.json']
                if t1:
                    ov.append('+t_plus_one=true')
                jobs.append(dict(name=name, n=n_eval, seed=s, overrides=ov))
    return jobs


def launch(job):
    out_json = RUNS / f'{job["name"]}.json'
    log = RUNS / f'{job["name"]}.log'
    if out_json.exists():
        return None, 'skip'
    cmd = [PY, '-u', str(ROOT / 'src' / 'e6_eval.py'),
           '--config-name=cube.yaml', 'policy=cube/weights.pt',
           f'eval.dataset_name={DATASET}',
           f'eval.num_eval={job["n"]}', f'seed={job["seed"]}',
           '+plan_config.history_len=3',
           'plan_config.action_block=5', 'solver.device=cuda',
           *job['overrides'], f'+out_json={out_json}']
    lf = open(log, 'w')
    p = subprocess.Popen(cmd, cwd=str(PLAN), stdout=lf, stderr=subprocess.STDOUT)
    return (p, lf, out_json, log), 'run'


STALL_SEC = 2700  # a single CEM solve tops out near 20 min; 45 min idle = wedged
MAX_RETRY = 2     # transient GL/CUDA crashes and wedges are retried


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['flat', 'probe'], required=True)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--t1', action='store_true',
                    help='probe only: T+1 ablation (finish event block)')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    RUNS.mkdir(exist_ok=True)
    # 100 envs, not 150: each live MuJoCo env holds a GLFW renderer, and at
    # n=150 the cube eval peaks at ~7.2 GB of 8 GB VRAM. The caching allocator
    # then crosses the VRAM ceiling mid-run and the FTZ device->host copy at
    # cem.py:271 stalls forever (GPU 99% util but 34 W). 100 envs peak ~5 GB.
    n_eval = 8 if args.smoke else 100
    if args.smoke:
        # keep smoke out of the real result namespace
        prefix = 'cube_smoke_' + args.phase
    else:
        prefix = None

    jobs = build_jobs(args.phase, n_eval, t1=args.t1)
    if args.limit:
        jobs = jobs[:args.limit]
    if prefix:
        for j in jobs:
            j['name'] = j['name'].replace('cube_' + args.phase, prefix, 1)
    print(f'[grid] phase={args.phase} jobs={len(jobs)} workers={args.workers} '
          f'n={n_eval}', flush=True)

    running, queue, done, failed = {}, list(jobs), 0, 0
    t_all = time.time()
    while queue or running:
        while queue and len(running) < args.workers:
            job = queue.pop(0)
            res, state = launch(job)
            if state == 'skip':
                done += 1
                print(f'[skip] {job["name"]}', flush=True)
                continue
            running[job['name']] = (res, job)
            print(f'[start] {job["name"]}  ({len(queue)} queued)', flush=True)
        time.sleep(10)
        now = time.time()
        for name in list(running):
            (p, lf, out_json, log), job = running[name]
            if p.poll() is None:
                # stall guard: no log growth for STALL_SEC -> kill + retry
                size = log.stat().st_size if log.exists() else 0
                mt = log.stat().st_mtime if log.exists() else now
                if job.get('_size') != size:
                    job['_size'] = size
                    job['_mt'] = mt
                elif now - job['_mt'] > STALL_SEC:
                    print(f'[STALL] {name}: log idle {int(now - job["_mt"])}s '
                          f'-> kill', flush=True)
                    p.kill()
                    p.wait()
                continue
            lf.close()
            del running[name]
            if p.returncode == 0 and out_json.exists():
                j = json.loads(out_json.read_text())
                done += 1
                print(f'[done ] {name}: {j["success_rate"]:.1f}% '
                      f'({int(sum(j["episode_successes"]))}/{job["n"]})',
                      flush=True)
                continue
            retries = job.get('retries', 0)
            tail = log.read_text(errors='replace')[-300:]
            print(f'[FAIL ] {name} rc={p.returncode} retry={retries}',
                  flush=True)
            if retries < MAX_RETRY:
                job['retries'] = retries + 1
                job.pop('_size', None)
                job.pop('_mt', None)
                queue.insert(0, job)
            else:
                failed += 1
                print(tail, flush=True)

    dt = (time.time() - t_all) / 3600
    print(f'\n[SUMMARY] done={done} failed={failed} in {dt:.2f}h')


if __name__ == '__main__':
    main()
