"""Regenerate dashboard data files from aiim2026/runs/*.json (all n=150 runs)."""
import json, math, glob, os
from datetime import datetime
from pathlib import Path

ROOT = Path(r"d:\Research-2026-jepa-boundary-detector")
RUNS = ROOT / "aiim2026" / "runs"
UPDATED = datetime.now().strftime("%H:%M:%S")


def load(pattern):
    out = {}
    for p in sorted(glob.glob(str(RUNS / pattern))):
        j = json.loads(Path(p).read_text(encoding="utf-8"))
        name = Path(p).stem
        out[name] = j
    return out


def mk_run(name, arm, seed, j):
    budget = j.get("eval_budget") or 0
    # receding*action_block estimate from arm string
    if "h5" in name:
        exec_len = 25
    elif "h10" in name:
        exec_len = 50
    elif "h15" in name:
        exec_len = 75
    else:
        exec_len = 50
    solves = max(1, math.ceil(budget / exec_len)) if budget else 0
    return {
        "name": name, "arm": arm, "seed": seed, "status": "done",
        "solves": solves, "expected": solves, "avg_s": None,
        "success": round(j["success_rate"], 2),
        "ep_n": len(j["episode_successes"]),
        "elapsed_min": None, "eta": "0h00m", "err": "",
    }


def write_js(var, runs, path):
    payload = {
        "updated": UPDATED, "elapsed": "—", "eta": "0h00m",
        "gpu": {"util": 0, "used": 0, "total": 8188},
        "overall": {"done": len(runs), "failed": 0, "total": len(runs), "pct": 100.0},
        "abort": None, "chain_done": True, "runs": runs,
    }
    Path(path).write_text(f"window.{var} = " + json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[write] {path}  ({len(runs)} runs)")


flat = load("flat_h*r*_d*_s*_n150.json")
probe = load("probe_t_hr_d*_s*_n150.json")

# ---- lane A: d75 decisive triangle (probe + flat 5/10/15 d75) ----
A = []
for name, j in flat.items():
    if j["goal_offset_steps"] != 75:
        continue
    h = "5,5" if "h5" in name else ("10,10" if "h10" in name else "15,15")
    A.append(mk_run(name, f"flat({h})", j["seed"], j))
for name, j in probe.items():
    if j["goal_offset_steps"] != 75:
        continue
    A.append(mk_run(name, "probe_t_hr", j["seed"], j))
A.sort(key=lambda r: (r["arm"], r["seed"]))
write_js("TASKA", A, ROOT / "taskA_data.js")

# ---- lane B: flat sweet-spot grid (all d25/d50/d75 flat) ----
B = []
for name, j in flat.items():
    h = "5,5" if "h5" in name else ("10,10" if "h10" in name else "15,15")
    B.append(mk_run(name, f"flat({h})·d{j['goal_offset_steps']}", j["seed"], j))
B.sort(key=lambda r: (int(r["arm"].split("·d")[1]), r["arm"], r["seed"]))
write_js("TASKB", B, ROOT / "taskB_data.js")

# ---- lane D: probe auto-adapt (d25/d50 probes) ----
D = []
for name, j in probe.items():
    if j["goal_offset_steps"] not in (25, 50):
        continue
    D.append(mk_run(name, f"probe_t_hr·d{j['goal_offset_steps']}", j["seed"], j))
D.sort(key=lambda r: (int(r["arm"].split("·d")[1]), r["seed"]))
write_js("TASKD", D, ROOT / "taskD_data.js")
