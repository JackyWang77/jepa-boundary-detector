"""Task D runner: probe_t_hr at d25/d50 — the missing probe trajectory.

Lane B showed the grid floor (flat h5/10/15 per d) but the probe arm only
exists at d75 (lane A). Claim (b) "auto-adapts horizon" is only testable
across d if we have probe at d25/d50 too — d25 is the sharpest test:
h5=85.6 vs h10=45.6 (40 pts apart). If probe lands near 85 it adapted;
near 45 it did not.

Usage: python run_taskD.py [seed]   # 42 / 43 / 44 -> one 2-run lane each;
without argument all 6 runs sequentially. Three seed lanes run in parallel
(static split, no work stealing, so no run is double-started). Each lane
writes its own taskD{seed}_data.js; the dashboard merges them.

Resume-safe (skip when out_json exists), aborts chain on first failure.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # aiim2026/
PLAN = ROOT / 'third_party' / 'stable-worldmodel' / 'scripts' / 'plan'
RUNS = ROOT / 'runs'
PY = r'C:\Users\Ghazz\miniconda3\envs\aiim\python.exe'

SOLVE_RE = re.compile(r'CEM solve time: ([\d.]+) seconds')

LANE_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else None
SEEDS = (LANE_SEED,) if LANE_SEED else (42, 43, 44)
DATA_JS = ROOT.parent / (f'taskD{LANE_SEED}_data.js' if LANE_SEED
                         else 'taskD_data.js')
JS_KEY = f'TASKD{LANE_SEED}' if LANE_SEED else 'TASKD'

# name, d(goal_offset), seed, expected solve LINES. A "CEM solve time" line
# is one batched replan over tens of envs, not one replan: lane A's d75
# probe run logged 27 lines total (~1159 replans / ~43 envs per batch).
# d25/d50 fire more often -> a few more lines; these are display hints.
RUN_SPECS = []
for seed in SEEDS:
    RUN_SPECS += [
        (f'probe_t_hr_d25_s{seed}_n150', 25, seed, 40, 2.5),
        (f'probe_t_hr_d50_s{seed}_n150', 50, seed, 35, 2.5),
    ]

T0 = time.time()
state = {
    'runs': [
        dict(name=n, d=d, seed=s, expected=x, fallback_solve=fb,
             status='queued', solves=0, solve_times=[], success=None,
             ep_n=None, started=None, ended=None, err='')
        for (n, d, s, x, fb) in RUN_SPECS
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


def fmt_hm(sec):
    if sec is None:
        return '—'
    sec = max(0, int(sec))
    return f'{sec // 3600}h{(sec % 3600) // 60:02d}m'


def eta(run):
    if run['status'] == 'done':
        return 0.0
    times = run['solve_times']
    if times:
        avg = sum(times) / len(times)
    else:
        done = [r for r in state['runs'] if r['solve_times']]
        avg = (sum(done[0]['solve_times']) / len(done[0]['solve_times'])
               if done else run['fallback_solve'])
    return max(0, run['expected'] - run['solves']) * avg


def write_data():
    now = time.time()
    runs_out = []
    for run in state['runs']:
        runs_out.append({
            'name': run['name'], 'arm': 'probe_t_hr', 'd': run['d'],
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
    tmp.write_text(f'window.{JS_KEY} = ' + json.dumps(data, ensure_ascii=False),
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
    os.environ.setdefault('STABLEWM_HOME', r'D:\stable_worldmodel')
    RUNS.mkdir(exist_ok=True)
    threading.Thread(target=monitor, daemon=True).start()
    for run in state['runs']:
        out_json = RUNS / f"{run['name']}.json"
        if out_json.exists():
            run['status'] = 'done'
            read_result(run)
            print(f"[skip] {run['name']} (json exists)", flush=True)
            continue
        cmd = [PY, '-u', str(ROOT / 'src' / 'e6_eval.py'),
               '--config-name=pusht.yaml', 'policy=pusht/weights.pt',
               f"eval.goal_offset_steps={run['d']}",
               'eval.eval_budget=150', 'eval.num_eval=150',
               '+plan_config.history_len=3',
               'plan_config.horizon=10', 'plan_config.receding_horizon=10',
               'plan_config.action_block=5', 'solver.device=cuda',
               f"seed={run['seed']}",
               f"+out_json={out_json}", '+arm=probe_t']
        run['status'] = 'running'
        run['started'] = time.time()
        print(f"[run ] {run['name']}", flush=True)
        log = RUNS / f"{run['name']}.log"
        with open(log, 'w') as f:
            rc = subprocess.run(cmd, cwd=PLAN, stdout=f,
                                stderr=subprocess.STDOUT).returncode
        run['ended'] = time.time()
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
