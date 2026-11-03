"""MRBP v4.0 - phase-conditioned multi-rate blocks (user's idea, 2026-10-20).

Design shift from v3.0, driven by three measured failures and one
literature theorem:
  - v3.0: tau ordering appeared (14.0/12.4/11.8 vs control 1.01) but the
    ratio failed C1 (1.19 < 2.5) and action summaries starved fast
    content (agent 0.35 -> 0.08).
  - E1: budget buys dimensionality, not organization; the content axis is
    action-determinism, confounded with timescale.
  - H-JEPA Thm 1 (arXiv 2610.06805): the objective selects by
    predictability; action-closed content is free to encode; barriers
    must act on the joint law across time.
  - E3: the task's OWN phase segmentation (dwell/stroke/maneuver) is a
    free ground truth.

v4.0 components:
  SEGMENTER (A2, data-driven): 1D k-means (K=3) on log per-step agent
    displacement -> soft phase labels, smoothed. No absolute thresholds;
    the split is decided by the data distribution. Phases only route
    HORIZONS - they are NOT fed to the encoder (else blocks could cheat
    by phase multiplexing instead of content placement).
  FEATURES (frozen, reused assets): e2jepa encoder (block 0.76) +
    bc encoder (agent 0.81) concatenated - slow and fast content each
    have a verified source. No training of perception; organization is
    the only learned thing.
  PHYSICS (informed inputs): causal-window scalars from the action
    stream and feature statistics (speed, |accel|, heading change,
    lag-1 and lag-4 feature energies) - content-level timescale cues,
    not phase labels.
  COMBINER g_theta + heads (A0/A3): 3 blocks x 32 dims, per-block
    LayerNorm; block k predicts ITS OWN future at horizon H[phi][k]:
    the PHASE moves the clock (maneuver shortens every block's
    horizon), the BLOCK sets the frequency ratio inside the phase.
    Fast phases get step-wise actions (v3.0's lesson); long horizons
    get coarsened action statistics (thermometer theorem, now applied
    only where it belongs).

Preregistered criteria (fixed before training):
  C1  per-block median tau ordered by design (slow >= mid >= fast)
      and slow/fast ratio >= 2.5
  C2  slow-block block-probe R2 minus fast-block >= 0.2, and fast-block
      agent-probe exceeds slow-block by >= 0.05
  C3  every block's mean dim std >= 0.01
  C4  phase-block co-activity: fast block is more active than slow in
      maneuver segments (>= 2x), and the slow block is calmer in dwell
      than its all-time baseline (< 0.8x)
  Control arm mrpb_v4_single: identical model/features/segmenter, but
      ONE horizon set {2,5} for every block and phase (single-rate).
      If it also passes, the clustering is a feature/phase artifact.

Run:  python -m e2jepa.mrbp_v4
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_cache
from .eval import PROBE_NPZ, encode_batch, ridge_r2
from .loss import gaussian_entropy, temperature
from .model import Encoder
from .tau_spectrum import tau_acf
from .train import device_auto

N_BLOCKS = 3
D_BLOCK = 32
STEPS = 3000
BATCH = 256
SEED = 0
OUT = os.path.join("results", "mrbp_v4.json")
PHASES = ["dwell", "stroke", "maneuver"]
# horizon table: H[phase][block] - the phase moves the clock,
# the block sets the frequency ratio inside the phase
HOR = {"dwell": [[10, 20], [5, 10], [2, 5]],
       "stroke": [[5, 10], [2, 5], [1, 2]],
       "maneuver": [[5, 10], [2, 5], [1, 2]]}
HOR_SINGLE = [[2, 5], [2, 5], [2, 5]]  # control arm: one clock
A_DIM = 15  # 5 flattened steps + mean(2) + last(2) + h_norm
TAU_RATIO = 2.5


def load_encoder(arm, device):
    """Frozen encoders from verified ckpts (e2jepa world model, bc policy)."""
    for name in ("world_model.pt", "policy.pt"):
        p = os.path.join("results", arm, name)
        if os.path.exists(p):
            ckpt = torch.load(p, map_location="cpu")
            if "enc" in ckpt:
                enc = Encoder().to(device)
                enc.load_state_dict(ckpt["enc"])
                enc.eval()
                return enc
    raise FileNotFoundError(f"no ckpt with enc for {arm}")


@torch.no_grad()
def extract_feats(obs, starts, ends, device):
    """Two-frame stacks for every frame; returns (N,256) concat feats."""
    encs = [load_encoder("e2jepa", device), load_encoder("bc", device)]
    idx, prev = [], []
    for e in range(len(starts)):
        s, en = int(starts[e]), int(ends[e])
        idx.extend(range(s, en))
        prev.extend([s] + list(range(s, en - 1)))
    stacks = np.stack([np.stack([obs[p], obs[i]], axis=0)
                       for p, i in zip(prev, idx)])
    feats = [encode_batch(enc, stacks, device) for enc in encs]
    return np.concatenate(feats, axis=1), np.asarray(idx)


def kmeans1d(x, k, iters=50, seed=0):
    c = np.quantile(x, np.linspace(0, 1, k + 2)[1:-1])
    for _ in range(iters):
        lab = np.abs(x[:, None] - c[None, :]).argmin(axis=1)
        newc = np.array([x[lab == j].mean() if (lab == j).any() else c[j]
                         for j in range(k)])
        if np.allclose(newc, c):
            break
        c = newc
    lab = np.abs(x[:, None] - c[None, :]).argmin(axis=1)
    return lab, np.sort(c)


def segment(agent, starts, ends):
    """Data-driven phases: k-means on log step displacement, smoothed.

    Phase ids: 0=dwell (slowest center) .. 2=maneuver (fastest). Labels
    are per-frame; smoothing uses a causal-majority window of 7 steps
    that never crosses episodes.
    """
    n = len(agent)
    speed = np.zeros(n)
    speed[:-1] = np.linalg.norm(np.diff(agent, axis=0), axis=1)
    lab_raw, centers = kmeans1d(np.log(speed + 1e-3), N_BLOCKS, seed=0)
    # order clusters by speed center: id 0 slowest (dwell)
    order = np.argsort(centers)
    lab = np.zeros(n, dtype=int)
    for new_id, old_id in enumerate(order):
        lab[lab_raw == old_id] = new_id
    smooth = lab.copy()
    for e in range(len(starts)):
        s, en = int(starts[e]), int(ends[e])
        for t in range(s, en):
            lo = max(s, t - 3)
            hi = min(en, t + 4)
            smooth[t] = np.bincount(lab[lo:hi],
                                    minlength=N_BLOCKS).argmax()
    return smooth


def phys_seq(feats, agent, starts, ends):
    """Per-frame physics scalars from the action stream + feature stats.

    (speed, |accel|, heading change, lag-1 energy, lag-4 energy) - all
    content statistics, no state labels, no phase labels. Computed
    within episodes only; boundary frames stay zero.
    """
    n = len(agent)
    phys = np.zeros((n, 5), dtype=np.float32)
    for e in range(len(starts)):
        s, en = int(starts[e]), int(ends[e])
        ag = agent[s:en]
        sp = np.zeros(en - s)
        sp[:-1] = np.linalg.norm(np.diff(ag, axis=0), axis=1)
        acc = np.zeros(en - s)
        acc[1:-1] = np.abs(sp[1:-1] - sp[:-2])
        d = np.diff(ag, axis=0)
        hd = np.zeros(en - s)
        cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
        hd[1:-1] = np.abs(cross) / (np.linalg.norm(d[:-1], axis=1)
                                    * np.linalg.norm(d[1:], axis=1) + 1e-9)
        ft = feats[s:en]
        e1 = np.zeros(en - s)
        e1[1:] = np.linalg.norm(np.diff(ft, axis=0), axis=1)
        e4 = np.zeros(en - s)
        e4[4:] = np.linalg.norm(ft[4:] - ft[:-4], axis=1)
        phys[s:en] = np.stack([sp, acc, hd, e1, e4], axis=1)
    return phys


class Combiner(torch.nn.Module):
    """g_theta: frozen feats + physics -> 3 blocks x 32, block-wise LN."""

    def __init__(self, d_in, d_phys=5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in + d_phys, 256), torch.nn.SiLU(),
            torch.nn.Linear(256, N_BLOCKS * D_BLOCK))
        self.lns = torch.nn.ModuleList(
            [torch.nn.LayerNorm(D_BLOCK) for _ in range(N_BLOCKS)])

    def forward(self, x):
        h = self.net(x)
        return torch.cat([ln(h[:, i * D_BLOCK:(i + 1) * D_BLOCK])
                          for i, ln in enumerate(self.lns)], dim=1)


class Head(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in + A_DIM, 128), torch.nn.LayerNorm(128),
            torch.nn.SiLU(), torch.nn.Linear(128, d_in))

    def forward(self, z_k, a_feat):
        return self.net(torch.cat([z_k, a_feat], dim=-1))


def action_feat(actions, idxs, hs, device):
    """Step-wise actions for short horizons, coarsened stats for long."""
    B = len(idxs)
    out = np.zeros((B, A_DIM), dtype=np.float32)
    for j, (i, h) in enumerate(zip(idxs, hs)):
        seg = actions[i:i + int(h)] / 512.0
        out[j, :min(10, len(seg) * 2)] = seg[:5].reshape(-1)[:10]
        out[j, 10:12] = seg.mean(axis=0)
        out[j, 12:14] = seg[-1]
        out[j, 14] = h / 20.0
    return torch.tensor(out, dtype=torch.float32, device=device)


def sample_windows(phases, starts, ends, rng, horizons):
    """Sample (t, h_k per block) with phase-routed horizons."""
    tabs = list(horizons.values()) if isinstance(horizons, dict) \
        else [horizons]
    max_h = max(h for tab in tabs for hs in tab for h in hs)
    idxs, hs = [], [[] for _ in range(N_BLOCKS)]
    while len(idxs) < BATCH:
        e = int(rng.integers(len(starts)))
        s, en = int(starts[e]), int(ends[e])
        if en - s < max_h + 2:
            continue
        t = int(rng.integers(s, en - max_h - 1))
        idxs.append(t)
        for k in range(N_BLOCKS):
            if isinstance(horizons, dict):
                hs_k = horizons[PHASES[phases[t]]][k]
            else:
                hs_k = horizons[k]
            hs[k].append(int(hs_k[rng.integers(len(hs_k))]))
    return np.asarray(idxs), [np.asarray(h) for h in hs]


def train_arm(arm, horizons, feats, phys, phases, actions, starts, ends,
              device, log):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    g = Combiner(feats.shape[1]).to(device)
    heads = torch.nn.ModuleList([Head(D_BLOCK)
                                 for _ in range(N_BLOCKS)]).to(device)
    opt = torch.optim.AdamW(list(g.parameters())
                            + list(heads.parameters()), lr=3e-4)
    ft = torch.tensor(feats, dtype=torch.float32, device=device)
    ph = torch.tensor(phys, dtype=torch.float32, device=device)
    step, t0 = 0, time.time()
    g.train()
    while step < STEPS:
        idxs, hs = sample_windows(phases, starts, ends, rng, horizons)
        x = torch.cat([ft[idxs], ph[idxs]], dim=1)
        z = g(x)
        T = temperature(step, STEPS)
        loss, rec = 0.0, {"arm": arm, "step": step, "u": [], "s": []}
        for k in range(N_BLOCKS):
            sl = slice(k * D_BLOCK, (k + 1) * D_BLOCK)
            a_feat = action_feat(actions, idxs, hs[k], device)
            ti = torch.tensor(idxs + hs[k], dtype=torch.long, device=device)
            with torch.no_grad():
                target = g(torch.cat([ft[ti], ph[ti]], dim=1))[:, sl]
            pred = heads[k](z[:, sl], a_feat)
            u = F.mse_loss(pred, target)
            s = gaussian_entropy(z[:, sl].cpu()).to(device)
            loss = loss + (u - T * s) / N_BLOCKS
            rec["u"].append(round(u.item(), 4))
            rec["s"].append(round(s.item(), 3))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 250 == 0:
            rec["sec"] = round(time.time() - t0, 1)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{arm}] step {step}: u={rec['u']} s={rec['s']}",
                  flush=True)
        step += 1
    return g


def measure_arm(g, feats, phys, phases, starts, ends, probe_npz,
                probe_x, device):
    """Block tau / probes / std / phase activity + criteria."""
    g.eval()
    ft = torch.tensor(feats, dtype=torch.float32, device=device)
    ph = torch.tensor(phys, dtype=torch.float32, device=device)
    with torch.no_grad():
        z_all = g(torch.cat([ft, ph], dim=1)).cpu().numpy()
    z_eps = [z_all[int(starts[e]):int(ends[e])]
             for e in range(len(starts))]
    tau, _ = tau_acf(z_eps)
    dnp = np.load(probe_npz)
    with torch.no_grad():
        zp = g(torch.tensor(probe_x, dtype=torch.float32,
                            device=device)).cpu().numpy()
    blocks = []
    for k in range(N_BLOCKS):
        idx = np.arange(k * D_BLOCK, (k + 1) * D_BLOCK)
        blocks.append({
            "tau_median": float(np.median(tau[idx])),
            "block_r2": float(ridge_r2(zp[:, idx], dnp["block"]).mean()),
            "agent_r2": float(ridge_r2(zp[:, idx], dnp["agent"]).mean()),
            "std": float(z_all[:, idx].std(axis=0).mean()),
        })
    # C4: phase-block co-activity
    act = np.zeros((N_BLOCKS, len(PHASES)))
    dzk = [np.linalg.norm(np.diff(z_all[:, k * D_BLOCK:(k + 1) * D_BLOCK],
                                  axis=0), axis=1)
           for k in range(N_BLOCKS)]
    ph_mid = phases[:-1]
    base = [float(dzk[k].mean()) for k in range(N_BLOCKS)]
    for p in range(len(PHASES)):
        m = ph_mid == p
        for k in range(N_BLOCKS):
            act[k, p] = float(dzk[k][m].mean())
    taus = [b["tau_median"] for b in blocks]
    crit = {
        "c1_tau_clusters": bool(taus[0] >= taus[1] >= taus[2]
                                and taus[0] / max(taus[2], 1e-9) >= TAU_RATIO),
        "c2_content_placement": bool(
            blocks[0]["block_r2"] - blocks[2]["block_r2"] >= 0.2
            and blocks[2]["agent_r2"] - blocks[0]["agent_r2"] >= 0.05),
        "c3_survival": bool(all(b["std"] >= 0.01 for b in blocks)),
        "c4_phase_activity": bool(
            act[2, 2] >= 2.0 * act[0, 2]
            and act[0, 0] <= 0.8 * base[0]),
    }
    return {"blocks": blocks, "tau_ratio": round(taus[0] / max(taus[2], 1e-9), 2),
            "block_taus": [round(t, 1) for t in taus],
            "activity": act.tolist(), "activity_base": base,
            "criteria": crit}


def main():
    device = device_auto()
    obs, actions, agent, starts, ends, _ = load_cache()
    print("[v4] extracting frozen dual-encoder features ...", flush=True)
    feats, _ = extract_feats(obs, starts, ends, device)
    print("[v4] segmenting action phases ...", flush=True)
    phases = segment(agent, starts, ends)
    phys = phys_seq(feats, agent, starts, ends)
    # probe-set features: same frozen encoders; frames form 60 ordered
    # rollouts of 120 steps each (env_utils.random_probe_set)
    dnp = np.load(PROBE_NPZ)
    enc_e2 = load_encoder("e2jepa", device)
    enc_bc = load_encoder("bc", device)
    probe_feats = np.concatenate(
        [encode_batch(enc, dnp["imgs"], device)
         for enc in (enc_e2, enc_bc)], axis=1)
    probe_agent = (dnp["agent"] + 0.5) * 512.0
    probe_phys = phys_seq(probe_feats, probe_agent,
                          np.arange(0, len(probe_agent), 120),
                          np.arange(120, len(probe_agent) + 1, 120))
    probe_x = np.concatenate([probe_feats, probe_phys], axis=1)

    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    log = open("results/mrbp_v4_metrics.jsonl", "a")
    for arm, horizons in [("mrpb_v4", HOR),
                          ("mrpb_v4_single", HOR_SINGLE)]:
        if arm in results and "criteria" in results.get(arm, {}):
            print(f"[{arm}] already done, skipping", flush=True)
            continue
        print(f"[{arm}] training {STEPS} steps ...", flush=True)
        g = train_arm(arm, horizons, feats, phys, phases, actions,
                      starts, ends, device, log)
        entry = measure_arm(g, feats, phys, phases, starts, ends,
                            PROBE_NPZ, probe_x, device)
        results[arm] = entry
        json.dump(results, open(OUT, "w"), indent=2)
        print(f"[{arm}] taus={entry['block_taus']} "
              f"ratio={entry['tau_ratio']} "
              f"criteria={entry['criteria']}", flush=True)
    log.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
