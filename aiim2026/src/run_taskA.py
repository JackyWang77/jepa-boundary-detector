"""Task A runner: decisive triangle on the 4060 (claim b, preregistered).

9 sequential e6_eval runs: flat(5,5) / flat(10,10) / probe_t_hr x seeds
42/43/44, d75, budget 150, n=150. Seed-major order so a full triangle
(seed 42) lands first for an early read.

Resume: a run is skipped when its out_json already exists (safe to relaunch
this script after a crash). Stops the chain at the first failed run so e.g.
an OOM does not burn the remaining budget.

A monitor thread tails each run's log (CEM solve lines), polls nvidia-smi
and rewrites taskA_data.js next to taskA_dashboard.html every 15 s.
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
DATA_JS = ROOT.parent / 'taskA_data.js'
PY = r'C:\Users\Ghazz\miniconda3\envs\aiim\python.exe'

SOLVE_RE = re.compile(r'CEM solve time: ([\d.]+) seconds')

# name, horizon, receding, seed, extra args, expected CEM solve lines
RUN_SPECS = []
for seed in (42, 43, 44):
    RUN_SPECS += [
        (f'flat_h5r5_d75_s{seed}_n150', 5, 5, seed, [], 6,
         'flat(5,5)', 120.0),
        (f'flat_h10r10_d75_s{seed}_n150', 10, 10, seed, [], 3,
         'flat(10,10)', 240.0),
        (f'probe_t_hr_d75_s{seed}_n150', 10, 10, seed, ['+arm=probe_t'],
         1500, 'probe_t_hr', 2.5),
    ]

T0 = time.time()
state = {
    'runs': [
        dict(name=n, horizon=h, receding=r, seed=s, extra=e,
             expected=x, arm=arm, fallback_solve=fb,
             status='queued', solves=0, solve_times=[], success=None,
             ep_n=None, started=None, ended=None, err='')
        for (n, h, r, s, e, x, arm, fb) in RUN_SPECS
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
        same = [r for r in state['runs']
                if r['arm'] == run['arm'] and r['solve_times']]
        avg = (sum(same[0]['solve_times']) / len(same[0]['solve_times'])
               if same else run['fallback_solve'])
    return max(0, run['expected'] - run['solves']) * avg


def write_data():
    now = time.time()
    runs_out = []
    for run in state['runs']:
        runs_out.append({
            'name': run['name'], 'arm': run['arm'], 'seed': run['seed'],
            'status': run['status'], 'solves': run['solves'],
            'expected': run['expected'],
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
    tmp.write_text('window.TASKA = ' + json.dumps(data, ensure_ascii=False),
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
               'eval.goal_offset_steps=75', 'eval.eval_budget=150',
               'eval.num_eval=150', '+plan_config.history_len=3',
               f"plan_config.horizon={run['horizon']}",
               f"plan_config.receding_horizon={run['receding']}",
               'plan_config.action_block=5', 'solver.device=cuda',
               f"seed={run['seed']}",
               f"+out_json={out_json}"] + run['extra']
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
        succ = run['success']
        print(f"[done] {run['name']} success={succ}", flush=True)
    state['chain_done'] = True
    write_data()
    print('[chain] all done', flush=True)


if __name__ == '__main__':
    main()
