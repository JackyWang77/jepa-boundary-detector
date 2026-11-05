"""Lane C: cube E1 flat baselines (claim-neutral, user-approved 2026-11-05).

9 sequential runs: flat(5,5) x d{25,50,75} x budget{50,100,150} x seeds
{42,43,44}, n=50/seed (PushT E1 protocol parity). Pure baseline data for
Table II + cube checkpoint validation; detector pipeline (E2/R2/E6) stays
frozen per the rhythm-gate outcome.

Same runner pattern as run_taskA/B: resume on existing json, stop on
failure, 15s heartbeat -> taskC_data.js, VRAM guard, MUJOCO_GL=glfw
(validated on this box during collection; e6_eval's setdefault 'egl'
is overridden by presetting the env var).
"""
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # aiim2026/
PLAN = ROOT / 'third_party' / 'stable-worldmodel' / 'scripts' / 'plan'
RUNS = ROOT / 'runs'
DATA_JS = ROOT.parent / 'taskC_data.js'
PY = r'C:\Users\Ghazz\miniconda3\envs\aiim\python.exe'

SOLVE_RE = re.compile(r'CEM solve time: ([\d.]+) seconds')
BUDGET = {25: 50, 50: 100, 75: 150}
MIN_FREE_MB = 2500

# name, d, seed, expected solve rounds
RUN_SPECS = [
    (f'cube_flat_d{d}_s{s}', d, s, BUDGET[d] // 25 + 1)
    for s in (42, 43, 44)
    for d in (25, 50, 75)
]

T0 = time.time()
state = {
    'runs': [
        dict(name=n, d=d, seed=s, expected=x, status='queued', solves=0,
             solve_times=[], success=None, ep_n=None, started=None, err='')
        for (n, d, s, x) in RUN_SPECS
    ],
    'gpu': {'util': None, 'used': None, 'total': None},
    'abort': None,
    'chain_done': False,
}


def read_result(run):
    js = RUNS / f"{run['name']}.json"
    if not js.exists():
        return
    try:
        d = json.loads(js.read_text())
        run['success'] = d.get('success_rate')
        run['ep_n'] = len(d.get('episode_successes', []))
    except (OSError, ValueError):
        pass


def scan_log(run):
    log = RUNS / f"{run['name']}.log"
    if not log.exists():
        return
    try:
        txt = log.read_text(errors='replace')
    except OSError:
        return
    times = [float(m) for m in SOLVE_RE.findall(txt)]
    run['solves'] = len(times)
    run['solve_times'] = times[-20:]


def poll_gpu():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,'
             'memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10).stdout.strip()
        u, w, t = [x.strip() for x in out.split(',')]
        state['gpu'] = {'util': int(u), 'used': int(w), 'total': int(t)}
    except Exception:
        pass


def wait_for_vram():
    while True:
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10).stdout.strip()
            free = int(out.split(',')[0])
        except Exception:
            return
        if free >= MIN_FREE_MB:
            return
        print(f'[vram] only {free}MB free < {MIN_FREE_MB}MB; wait 60s',
              flush=True)
        time.sleep(60)


def fmt_hm(sec):
    if sec is None:
        return '—'
    sec = max(0, int(sec))
    return f'{sec // 3600}h{(sec % 3600) // 60:02d}m'


def eta(run):
    if run['status'] == 'done':
        return 0.0
    times = run['solve_times']
    avg = (sum(times) / len(times)) if times else 90.0
    return max(0, run['expected'] - run['solves']) * avg


def write_data():
    now = time.time()
    runs_out = []
    for run in state['runs']:
        runs_out.append({
            'name': run['name'], 'arm': f"flat(5,5)·d{run['d']}",
            'seed': run['seed'], 'status': run['status'],
            'solves': run['solves'], 'expected': run['expected'],
            'avg_s': (round(sum(run['solve_times']) / len(run['solve_times']), 1)
                      if run['solve_times'] else None),
            'success': run['success'], 'ep_n': run['ep_n'],
            'elapsed_min': (round((now - run['started']) / 60, 1)
                            if run['started'] else None),
            'eta': fmt_hm(eta(run)),
            'err': run['err'][-800:],
        })
    done = sum(r['status'] == 'done' for r in state['runs'])
    failed = sum(r['status'] == 'failed' for r in state['runs'])
    data = {
        'updated': datetime.now().strftime('%H:%M:%S'),
        'elapsed': fmt_hm(now - T0),
        'eta': fmt_hm(sum(eta(r) for r in state['runs'])),
        'gpu': state['gpu'],
        'overall': {'done': done, 'failed': failed,
                    'total': len(state['runs']),
                    'pct': round(100 * (done + failed)
                                 / len(state['runs']), 1)},
        'abort': state['abort'],
        'chain_done': state['chain_done'],
        'runs': runs_out,
    }
    tmp = DATA_JS.with_suffix('.tmp')
    tmp.write_text('window.TASKC = ' + json.dumps(data, ensure_ascii=False),
                   encoding='utf-8')
    os.replace(tmp, DATA_JS)


def monitor():
    while True:
        for run in state['runs']:
            scan_log(run)
            if run['status'] in ('done', 'running'):
                read_result(run)
        poll_gpu()
        try:
            write_data()
        except OSError:
            pass
        time.sleep(15)


def main():
    os.environ['STABLEWM_HOME'] = r'D:\stable_worldmodel'
    os.environ['MUJOCO_GL'] = 'glfw'
    RUNS.mkdir(exist_ok=True)
    threading.Thread(target=monitor, daemon=True).start()
    for run in state['runs']:
        out_json = RUNS / f"{run['name']}.json"
        if out_json.exists():
            run['status'] = 'done'
            read_result(run)
            print(f"[skip] {run['name']} (json exists)", flush=True)
            continue
        wait_for_vram()
        cmd = [PY, '-u', str(ROOT / 'src' / 'e6_eval.py'),
               '--config-name=cube.yaml', 'policy=cube/weights.pt',
               f"eval.goal_offset_steps={run['d']}",
               f"eval.eval_budget={BUDGET[run['d']]}",
               'eval.num_eval=50', '+plan_config.history_len=3',
               'plan_config.horizon=5', 'plan_config.receding_horizon=5',
               'plan_config.action_block=5', 'solver.device=cuda',
               f"seed={run['seed']}",
               f"+out_json={out_json}"]
        run['status'] = 'running'
        run['started'] = time.time()
        print(f"[run ] {run['name']}", flush=True)
        log = RUNS / f"{run['name']}.log"
        with open(log, 'w') as f:
            rc = subprocess.run(cmd, cwd=PLAN, stdout=f,
                                stderr=subprocess.STDOUT).returncode
        if rc != 0 or not out_json.exists():
            run['status'] = 'failed'
            try:
                run['err'] = log.read_text(errors='replace')[-2000:]
            except OSError:
                pass
            state['abort'] = run['name']
            write_data()
            print(f"[FAIL] {run['name']} rc={rc} -> stop chain", flush=True)
            break
        run['status'] = 'done'
        read_result(run)
        print(f"[done] {run['name']} success={run['success']}", flush=True)
    state['chain_done'] = True
    write_data()
    print('[chain] all done', flush=True)


if __name__ == '__main__':
    main()
