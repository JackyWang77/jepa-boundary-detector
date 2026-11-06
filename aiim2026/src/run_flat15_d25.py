"""Run flat(15,15)·d25 seed43+44 individually."""
import subprocess, sys, os
import re, time, json, numpy as np
from pathlib import Path

os.environ.setdefault('STABLEWM_HOME', r'D:\stable_worldmodel')

ROOT = Path(r"d:\Research-2026-jepa-boundary-detector\aiim2026")
PLAN = ROOT / "third_party" / "stable-worldmodel" / "scripts" / "plan"
PY = r"C:\Users\Ghazz\miniconda3\envs\aiim\python.exe"
OUT = ROOT / "runs"

SOLVE_RE = re.compile(r'CEM solve time: ([\d.]+) seconds')
N = 150; D = 25; H = 15; R = 15; BUDGET = 50

for seed in [43, 44]:
    name = f"flat_h15r15_d25_s{seed}_n{N}"
    out_json = OUT / f"{name}.json"
    if out_json.exists():
        j = json.loads(out_json.read_text())
        print(f"[skip] {name}: sr={j['success_rate']:.1%}")
        continue

    cmd = [PY, "-u", str(ROOT / "src" / "e6_eval.py"),
           "--config-name=pusht.yaml", "policy=pusht/weights.pt",
           f"eval.goal_offset_steps={D}",
           f"eval.eval_budget={BUDGET}",
           f"eval.num_eval={N}", "+plan_config.history_len=3",
           f"plan_config.horizon={H}",
           f"plan_config.receding_horizon={R}",
           "plan_config.action_block=5", "solver.device=cuda",
           f"seed={seed}",
           f"+out_json={out_json}"]

    log_path = OUT / f"{name}.log"
    t0 = time.time()
    print(f"[run] {name}...", flush=True)
    with open(log_path, 'w') as lf:
        rc = subprocess.run(cmd, cwd=str(PLAN), stdout=lf, stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0

    if rc != 0:
        print(f"[FAIL] {name} exit={rc} log_tail: {log_path.read_text(errors='replace')[-500:]}")
        continue

    j = json.loads(out_json.read_text())
    print(f"[done] {name}: sr={j['success_rate']:.1%} ({int(sum(j['episode_successes']))}/{N})  time={dt/60:.0f}min")

# summary
print(f"\n{'='*50}")
print("SUMMARY flat(15,15)·d25")
print(f"{'='*50}")
srs = []
for seed in [42, 43, 44]:
    jp = OUT / f"flat_h15r15_d25_s{seed}_n{N}.json"
    if jp.exists():
        j = json.loads(jp.read_text())
        srs.append((seed, j['success_rate']))
        print(f"  s{seed}: {j['success_rate']:.1%}")
if srs:
    vals = [v for _, v in srs]
    print(f"  mean={np.mean(vals):.1%}  [{np.min(vals):.1%}-{np.max(vals):.1%}]")

print(f"\nCOMPARE d25 grid (n=450 mean):")
print(f"  flat(5,5)    = 85.6%")
print(f"  flat(10,10)  = 45.6%")
if srs:
    print(f"  flat(15,15)  = {np.mean(vals)*100:.1f}%  <-- NEW (plan-once, no replan in budget)")
