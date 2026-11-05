"""Task B runner: PushT sweet-spot grid (claim b2, preregistered) on the 4060.

18 sequential e6_eval runs filling the argmax grid over flat horizons:
  d25 (budget 50):  flat(5,5), flat(10,10)   [flat(15,15) impossible: 15*5=75 > 50]
  d50 (budget 100): flat(5,5), flat(10,10), flat(15,15)
  d75 (budget 150): flat(15,15)              [{5,10} arms come from Task A]
x seeds 42/43/44, n=150, seed-major so each seed completes a full grid.

Same resume/monitor pattern as run_taskA.py, plus a free-VRAM guard before
each run (lanes A/B/C share the 8GB GPU). Writes taskB_data.js for the
unified dashboard.
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
DATA_JS = ROOT.parent / 'taskB_data.js'
PY = r'C:\Users\Ghazz\miniconda3\envs\aiim\python.exe'

SOLVE_RE = re.compile(r'CEM solve time: ([\d.]+) seconds')
BUDGET = {25: 50, 50: 100, 75: 150}
MIN_FREE_MB = 2500

# name, d, horizon, receding, seed, expected solve rounds, label, fallback s
RUN_SPECS = []
for seed in (42, 43, 44):
    RUN_SPECS += [
        (f'flat_h15r15_d75_s{seed}_n150', 75, 15, 15, seed, 3,
         'flat(15,15)·d75', 360.0),
        (f'flat_h15r15_d50_s{seed}_n150', 50, 15, 15, seed, 2,
         'flat(15,15)·d50', 240.0),
        (f'flat_h10r10_d50_s{seed}_n150', 50, 10, 10, seed, 3,
         'flat(10,10)·d50', 240.0),
        (f'flat_h5r5_d50_s{seed}_n150', 50, 5, 5, seed, 5,
         'flat(5,5)·d50', 120.0),
        (f'flat_h10r10_d25_s{seed}_n150', 25, 10, 10, seed, 2,
         'flat(10,10)·d25', 120.0),
        (f'flat_h5r5_d25_s{seed}_n150', 25, 5, 5, seed, 3,
         'flat(5,5)·d25', 60.0),
    ]

T0 = time.time()
state = {
    'runs': [
        dict(name=n, d=d, horizon=h, receding=r, seed=s, expected=x,
             arm=lab, fallback_solve=fb, status='queued', solves=0,
             solve_times=[], success=None, ep_n=None, started=None,
             err='')
        for (n, d, h, r, s, x, lab, fb) in RUN_SPECS
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
    tmp.write_text('window.TASKB = ' + json.dumps(data, ensure_ascii=False),
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
        wait_for_vram()
        cmd = [PY, '-u', str(ROOT / 'src' / 'e6_eval.py'),
               '--config-name=pusht.yaml', 'policy=pusht/weights.pt',
               f"eval.goal_offset_steps={run['d']}",
               f"eval.eval_budget={BUDGET[run['d']]}",
               'eval.num_eval=150', '+plan_config.history_len=3',
               f"plan_config.horizon={run['horizon']}",
               f"plan_config.receding_horizon={run['receding']}",
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
