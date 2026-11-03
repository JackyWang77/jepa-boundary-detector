"""E3: the task's own hierarchy — ground truth for prediction P3.

PushT teleop data contains three natural timescales (theory step 1: the
environment's transfer operator has clustered eigenvalues):
  waypoint dwell  — block still (quasi-static slow modes, tau -> large)
  stroke          — block being pushed (intermediate, tau ~ stroke length)
  agent maneuver  — hand repositioning (fast, tau ~ few steps)

We measure them with the SAME estimator A used for the latent audit
(tau_acf on raw state variables), plus a stroke/dwell segmentation from
block-mask centroid motion. Output: results/task_hierarchy.json.

Block tracking: the T-block is the blue-gray blob (agent is a much more
saturated blue circle); frames are deterministic renders, so a still block
gives exactly zero centroid displacement — the move/still threshold is
trivially separable (verified by the printed histogram).

Reference conventions (verified against gym_pusht source, see verify.py):
the mask CENTROID differs from the env's block_pose by a constant offset
that rotates with the block (~4.46 px at 56-res; block_pose is the tee's
bottom-center, the centroid its visual middle). E3 uses displacements
and tau only, both invariant to that offset. The second-moment
orientation is the T's stem axis, ~90 deg away from block_pose's angle
reference — recorded but never used downstream.
"""
import json
import os

import numpy as np

from .data import load_cache
from .tau_spectrum import tau_acf

BLOCK_COLOR = np.array([150, 170, 185])
TOL = 35
WINDOW = 5  # cumulative-displacement window (steps)
# block px over WINDOW at 56x56: pushes accumulate coherently,
# per-step resize jitter does not
BLOCK_MOVE_THR = 0.8
AGENT_MOVE_THR = 0.3     # agent px per step at 512 scale (parquet is exact)
MIN_DWELL = 3
OUT = os.path.join("results", "task_hierarchy.json")


def block_pose(frame):
    """Blue-gray mask centroid + orientation (cos, sin) via second moments."""
    f = frame.astype(int)
    m = np.abs(f - BLOCK_COLOR).max(axis=-1) < TOL
    if m.sum() < 12:
        return None
    ys, xs = np.where(m)
    cx, cy = xs.mean(), ys.mean()
    x = xs - cx
    y = ys - cy
    cov = np.array([[(x * x).mean(), (x * y).mean()],
                    [(x * y).mean(), (y * y).mean()]])
    _, v = np.linalg.eigh(cov)
    d = v[:, -1]
    ang = np.arctan2(d[1], d[0])
    return cx, cy, float(np.cos(ang)), float(np.sin(ang))


def segments(moving):
    """Runs of True/False in a bool array -> list of (value, length)."""
    out, cur, n = [], bool(moving[0]), 1
    for v in moving[1:]:
        if bool(v) == cur:
            n += 1
        else:
            out.append((cur, n))
            cur, n = bool(v), 1
    out.append((cur, n))
    return out


def main():
    obs, _, agent, starts, ends, _ = load_cache()
    poses, agent_eps, strokes_ep, dwells_ep, maneuvers_ep = [], [], [], [], []
    disp_hist = np.zeros(20)
    for e in range(len(starts)):
        s, en = int(starts[e]), int(ends[e])
        P, last = [], None
        for i in range(s, en):
            p = block_pose(obs[i])
            if p is not None:
                last = p
            P.append(last)
        keep = [i for i, p in enumerate(P) if p is not None]
        if len(keep) < 10:
            continue
        pts = np.array([P[i] for i in keep])
        poses.append(pts)
        ag = agent[s:en][keep]
        agent_eps.append(ag / 512.0)
        # block strokes: cumulative displacement over a sliding window
        c = pts[:, :2]
        moved = np.zeros(len(c), dtype=bool)
        for i in range(len(c) - WINDOW):
            if np.linalg.norm(c[i + WINDOW] - c[i]) > BLOCK_MOVE_THR:
                moved[i:i + WINDOW + 1] = True
        for x in np.linalg.norm(np.diff(c, axis=0), axis=1):
            disp_hist[min(int(x / 0.25), 19)] += 1
        segs = segments(moved)
        strokes_ep.append([n for v, n in segs if v])
        dwells_ep.append([n for v, n in segs if not v and n >= MIN_DWELL])
        # agent maneuvers: runs of per-step motion from exact parquet state
        am = np.linalg.norm(np.diff(ag, axis=0), axis=1) > AGENT_MOVE_THR
        maneuvers_ep.append([n for v, n in segments(am) if v])

    def med_tau(eps_list):
        tau, _ = tau_acf(eps_list, dmax=96)
        return float(np.median(tau))

    all_strokes = [x for ep in strokes_ep for x in ep]
    all_dwells = [x for ep in dwells_ep for x in ep]
    all_maneuvers = [x for ep in maneuvers_ep for x in ep]
    n_move = sum(len(ep) for ep in strokes_ep)
    n_total = sum(len(p) - 1 for p in poses)
    res = {
        "n_episodes": len(poses),
        "strokes_per_ep_median": float(np.median([len(ep)
                                                  for ep in strokes_ep])),
        "stroke_len_quartiles": [float(np.quantile(all_strokes, q))
                                 for q in (.1, .25, .5, .75, .9)],
        "dwell_len_quartiles": [float(np.quantile(all_dwells, q))
                                for q in (.1, .25, .5, .75, .9)]
        if all_dwells else [],
        "maneuver_len_quartiles": [float(np.quantile(all_maneuvers, q))
                                   for q in (.1, .25, .5, .75, .9)]
        if all_maneuvers else [],
        "tau_block_x": med_tau([p[:, 0:1] for p in poses]),
        "tau_block_y": med_tau([p[:, 1:2] for p in poses]),
        "tau_agent_x": med_tau([a[:, 0:1] for a in agent_eps]),
        "tau_agent_y": med_tau([a[:, 1:2] for a in agent_eps]),
        "moving_frac": float(n_move / max(n_total, 1)),
        "disp_histogram_0.25px_bins": disp_hist.tolist(),
    }
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
