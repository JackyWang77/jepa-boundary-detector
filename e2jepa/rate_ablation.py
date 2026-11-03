"""E1: rate ablation - does information condense in tau order? (P1)

Theory (coding theorem, step 2 of the derivation): under a rate-limited
predictive objective slow content is cheap to keep, fast content is
expensive - raising the entropy budget should admit block pose FIRST and
agent position LATER, like cooling a mixed vapor: components condense in
boiling-point order. The sweep tests that order directly.

Preregistered predictions (fixed before any run):
  P1a  early saturation: at the LOWEST surviving budget, block probe R2
       already reaches >= 0.8 * (its max over the sweep)
  P1b  late, monotone entry for agent: R2_agent is non-decreasing across
       surviving budgets, and its entry point (R2 >= 0.3) sits at a
       strictly higher budget than block's entry point
  P1c  a collapse boundary exists: below some budget the survival guard
       fails (mean dim std < 0.01, the measured energy_only regime)
  Falsifier: if block and agent enter together (entry points within one
  sweep step), the rate-vs-timescale story is wrong - recorded either way.

Protocol: e2jepa's exact world-model training (architecture, 3000 steps,
batch 256, seed 0, EMA, cooling shape) with the temperature schedule
scaled: T(step) = t_max * temperature(step, steps) with the original
cosine 1.0 -> 0.05 shape, so the peak is t_max and the floor 0.05*t_max.
t_max = 1.0 replicates the released e2jepa arm (consistency check).
Sweep: 0.03, 0.1, 0.3, 1.0, 3.0. No policy head is trained: the question
is what the WORLD MODEL keeps, not what BC can squeeze out.

Measurements per point (aligned probe set, verified stack):
  block probe R2 (x, y, cos, sin), agent probe R2 (x, y),
  mean dim std + participation ratio (survival/information),
  median data-side tau of the latents.
Output: results/rate_ablation.json (+ .png), merge-safe.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import WindowDataset, collate_windows, load_cache
from .eval import PROBE_NPZ, encode_batch, ridge_r2
from .loss import temperature
from .model import Encoder, Predictor, emma_update, make_target, rollout_latents
from .tau_spectrum import encode_episodes, participation, tau_acf
from .train import entropy_on_cpu

SWEEP = [0.03, 0.10, 0.30, 1.0, 3.0]
STEPS = 3000
BATCH = 256
SEED = 0
OUT = os.path.join("results", "rate_ablation.json")


def train_point(t_max, device, log):
    """One e2jepa world-model run with the scaled temperature schedule."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    enc = Encoder().to(device)
    pred = Predictor().to(device)
    tgt_enc = make_target(enc).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=3e-4)
    ds = WindowDataset(k_max=4, n_samples=STEPS * BATCH, seed=SEED)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    collate_fn=collate_windows, num_workers=0)
    step, t0 = 0, time.time()
    enc.train()
    while step < STEPS:
        for x, a, mask, y, ks in dl:
            if step >= STEPS:
                break
            x, a, mask, y = (x.to(device), a.to(device),
                             mask.to(device), y.to(device))
            z = enc(x)
            with torch.no_grad():
                zt = tgt_enc(y)
            preds = rollout_latents(pred, z, a, int(ks.max()))
            u = 0.0
            for i, zp in enumerate(preds):
                m = mask[:, i] > 0
                if m.any():
                    u = u + F.mse_loss(zp[m], zt[m])
            u = u / len(preds)
            T = t_max * temperature(step, STEPS)
            loss = u - T * entropy_on_cpu(z, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            emma_update(enc, tgt_enc, momentum=0.999)
            if step % 250 == 0:
                log.write(json.dumps(
                    {"t_max": t_max, "step": step, "u": round(u.item(), 4),
                     "T": round(T, 4), "loss": round(loss.item(), 4),
                     "sec": round(time.time() - t0, 1)}) + "\n")
                log.flush()
            step += 1
    return enc


def measure(enc, obs, starts, ends, device):
    dnp = np.load(PROBE_NPZ)
    n = 4000
    Z = encode_batch(enc, dnp["imgs"][:n], device)
    r2_block = float(ridge_r2(Z, dnp["block"][:n]).mean())
    r2_agent = float(ridge_r2(Z, dnp["agent"][:n]).mean())
    z_eps = encode_episodes(enc, obs, starts, ends, device)
    dim_std = float(np.concatenate(z_eps).std(axis=0).mean())
    tau, _ = tau_acf(z_eps)
    return {"block_r2": r2_block, "agent_r2": r2_agent,
            "mean_dim_std": dim_std,
            "participation_ratio": participation(z_eps),
            "tau_median": float(np.median(tau))}


def entry_point(results, target, thr=0.3):
    """Lowest t_max whose target R2 >= thr, among surviving points."""
    keys = [t for t in results if not t.startswith("_")]
    for t in sorted(keys, key=float):
        e = results[t]
        if e["mean_dim_std"] < 0.01:
            continue
        if e[target] >= thr:
            return float(t)
    return None


def score(results):
    alive = {t: e for t, e in results.items()
             if not t.startswith("_") and e["mean_dim_std"] >= 0.01}
    if not alive:
        return {"p1a_block_early": False, "p1b_agent_late_monotone": False,
                "p1c_collapse_boundary": True, "entry_block": None,
                "entry_agent": None, "agent_monotone": False, "n_alive": 0}
    ts = sorted(alive, key=float)
    blocks = [alive[t]["block_r2"] for t in ts]
    agents = [alive[t]["agent_r2"] for t in ts]
    low_t = ts[0]
    p1a = bool(alive[low_t]["block_r2"] >= 0.8 * max(blocks))
    p1b_mono = all(agents[i] <= agents[i + 1] + 0.02
                   for i in range(len(agents) - 1))
    e_block = entry_point(results, "block_r2")
    e_agent = entry_point(results, "agent_r2")
    both = bool(e_block is not None and e_agent is not None)
    p1b = bool(p1b_mono and both and e_agent > e_block)
    p1c = bool(any(e["mean_dim_std"] < 0.01 for e in results.values()))
    return {"p1a_block_early": p1a, "p1b_agent_late_monotone": p1b,
            "p1c_collapse_boundary": p1c,
            "entry_block": e_block, "entry_agent": e_agent,
            "agent_monotone": p1b_mono,
            "n_alive": len(alive)}


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = sorted((float(t) for t in results if not t.startswith("_")))
    alive = [results[str(t)]["mean_dim_std"] >= 0.01 for t in ts]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(ts, [results[str(t)]["block_r2"] for t in ts], "o-",
            label="block (slow)")
    ax.plot(ts, [results[str(t)]["agent_r2"] for t in ts], "s-",
            label="agent (fast)")
    for t, ok in zip(ts, alive):
        if not ok:
            ax.annotate("collapsed", (t, 0.02), fontsize=7, color="red",
                        ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("temperature budget t_max (log)")
    ax.set_ylabel("linear probe R^2")
    ax.set_title("condensation order: rate sweep (P1)")
    ax.legend()

    ax = axes[1]
    ax.plot(ts, [results[str(t)]["mean_dim_std"] for t in ts], "o-",
            label="dim std")
    ax.plot(ts, [results[str(t)]["participation_ratio"] for t in ts], "s-",
            label="participation ratio")
    ax.axhline(0.01, color="red", ls=":", lw=1, label="collapse guard")
    ax.set_xscale("log")
    ax.set_xlabel("temperature budget t_max (log)")
    ax.set_title("survival / information content")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("results/rate_ablation.png", dpi=140)
    plt.close()


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() \
        else torch.device("cpu")
    obs, _, _, starts, ends, _ = load_cache()
    results = {}
    if os.path.exists(OUT):
        results = json.load(open(OUT))
    log = open("results/rate_ablation_metrics.jsonl", "a")
    for t_max in SWEEP:
        key = str(t_max)
        if key in results and len(results[key]) == 5:
            print(f"[rate:{key}] already done, skipping", flush=True)
            continue
        print(f"[rate:{key}] training {STEPS} steps ...", flush=True)
        enc = train_point(t_max, device, log)
        enc.eval()
        results[key] = measure(enc, obs, starts, ends, device)
        print(f"[rate:{key}] block={results[key]['block_r2']:.3f} "
              f"agent={results[key]['agent_r2']:.3f} "
              f"std={results[key]['mean_dim_std']:.3f} "
              f"PR={results[key]['participation_ratio']:.1f} "
              f"tau_med={results[key]['tau_median']:.1f}", flush=True)
        json.dump(results, open(OUT, "w"), indent=2)  # incremental save
    log.close()
    sweep = {k: v for k, v in results.items() if not k.startswith("_")}
    crit = score(sweep)
    results["_criteria"] = crit
    json.dump(results, open(OUT, "w"), indent=2)
    plot(sweep)
    print(f"[rate] criteria={crit}", flush=True)
    print("wrote results/rate_ablation.json and rate_ablation.png")


if __name__ == "__main__":
    main()
