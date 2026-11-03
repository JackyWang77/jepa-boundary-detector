"""Tau spectrum: how long does each latent dimension stay true?

Two independent estimators of the characteristic time tau
(research_story.md section 3):
  A (data side)  per-dim autocorrelation of z(t) along teleop episodes;
                 tau_i = sum_{D<=D*} C_i(D), truncated at the first
                 non-positive lag (initial-positive-sequence rule).
  B (model side) eigenvalues of the learned one-step latent dynamics
                 linearized at data points: lambda -> tau = -1/ln|lambda|.

A uses no model, B uses no labels; agreement of the sorted spectra means
the time-scale structure is real, not an artifact of either side.

Predictions tested (research_story.md section 4):
  1. slow dims (by tau) linearly decode block pose, fast dims decode
     agent position - for prediction-trained encoders; reversed or
     absent for task-trained (bc) and collapsed (energy_only) ones.
  2. energy_only spectrum degenerates (variance collapse -> flat tau).
  3. the largest spectral gap is measured, not tuned: it says where a
     hierarchy boundary belongs (criterion 1).
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.func import jacrev, vmap

from .data import load_cache
from .eval import PROBE_NPZ, encode_batch, load_policy, ridge_r2
from .model import Predictor
from .train import ARMS, BAND_MULT, N_BANDS, device_auto

DMAX = 96            # max ACF lag (episode lengths: min 49, median 122)
JAC_POINTS = 200     # trajectory points for the Jacobian estimator
JAC_TAU_CAP = 1000   # storage cap for -1/ln(lambda) when lambda ~ 1


def episode_stacks(obs, s, e):
    """(L,2,56,56,3) uint8; frame t pairs (obs[t-1], obs[t]) clamped at s."""
    prev = np.concatenate([obs[s:s + 1], obs[s:e - 1]])
    return np.stack([prev, obs[s:e]], axis=1)


def encode_episodes(enc, obs, starts, ends, device):
    """Encode every teleop frame; returns list of (L,d) arrays per episode."""
    out, buf, lens = [], [], []
    for e in range(len(starts)):
        s, en = int(starts[e]), int(ends[e])
        buf.append(episode_stacks(obs, s, en))
        lens.append(en - s)
        if sum(lens) >= 1024 or e == len(starts) - 1:
            zs = encode_batch(enc, np.concatenate(buf), device)
            i = 0
            for ln in lens:
                out.append(zs[i:i + ln])
                i += ln
            buf, lens = [], []
    return out


def tau_acf(z_eps, dmax=DMAX):
    """Pooled per-dim autocorrelation; ACF pairs never cross episodes.

    Each episode is demeaned; lag D is weighted by its pair count. The
    integral stops at the first non-positive lag (initial positive
    sequence) so noise at large lags cannot cancel real signal.
    Returns (tau[d], censored[d]).
    """
    d = z_eps[0].shape[1]
    num = np.zeros((d, dmax + 1))
    den = np.zeros((d, dmax + 1))
    for z in z_eps:
        zdm = z - z.mean(axis=0, keepdims=True)
        ms = (zdm * zdm).sum(axis=0)
        L = len(zdm)
        for dl in range(1, min(dmax, L - 1) + 1):
            num[:, dl] += (zdm[:-dl] * zdm[dl:]).sum(axis=0)
            den[:, dl] += (L - dl) * ms / L
    C = np.ones((d, dmax + 1))
    nz = den > 0
    C[nz] = (num / np.maximum(den, 1e-12))[nz]
    tau = np.ones(d)
    censored = np.zeros(d, dtype=bool)
    for i in range(d):
        neg = np.nonzero(C[i, 1:] <= 0)[0]
        k = int(neg[0]) + 1 if len(neg) else dmax + 1
        tau[i] = C[i, :k].sum()
        censored[i] = len(neg) == 0
    return tau, censored


def participation(z_eps):
    """Effective dimensionality of the latent covariance (1 = collapsed)."""
    Z = np.concatenate(z_eps)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    ev = np.linalg.eigvalsh(Zc.T @ Zc / len(Z))
    ev = np.maximum(ev, 0.0)
    return float(ev.sum() ** 2 / np.sum(ev * ev))


def largest_gap(ts, lo=8):
    """Largest multiplicative drop between consecutive sorted taus."""
    hi = len(ts) - 8
    best_k, best_r = lo, 0.0
    for k in range(lo, hi):
        r = ts[k - 1] / max(ts[k], 1e-9)
        if r > best_r:
            best_k, best_r = k, r
    return {"after_rank": best_k, "ratio": round(float(best_r), 3),
            "tau_above": float(ts[best_k - 1]),
            "tau_below": float(ts[best_k])}


def tau_jacobian(arm, obs, actions, starts, ends, enc, device,
                 n_points=JAC_POINTS, seed=3, out_root="results"):
    """Estimator B: eigen-spectrum of J = df/dz at held-out data points."""
    ckpt = torch.load(os.path.join(out_root, arm, "world_model.pt"),
                      map_location="cpu")
    pred = Predictor()
    pred.load_state_dict(ckpt["pred"])
    pred.eval()
    rng = np.random.default_rng(seed)
    val = [i for i in range(len(starts)) if i % 10 == 9]
    frames, acts = [], []
    for _ in range(n_points):
        e = val[int(rng.integers(len(val)))]
        s, en = int(starts[e]), int(ends[e])
        t = int(rng.integers(s, en - 1))
        frames.append(episode_stacks(obs, s, en)[t - s])
        acts.append(actions[t] / 512.0)
    zs = torch.tensor(encode_batch(enc, np.stack(frames), device))
    aa = torch.tensor(np.stack(acts), dtype=torch.float32)

    def f(z, a):
        return pred(z.unsqueeze(0), a.unsqueeze(0)).squeeze(0)

    J = vmap(jacrev(f, argnums=0))(zs, aa).detach().numpy()  # (n, d, d)
    lams = np.sort(np.abs(np.linalg.eigvals(J)), axis=1)[:, ::-1]
    d = lams.shape[1]
    tau_rank = np.zeros(d)
    frac_unstable = np.zeros(d)
    for j in range(d):
        col = lams[:, j]
        stable = col[col < 1.0]
        frac_unstable[j] = float((col >= 1.0).mean())
        g = (float(np.exp(np.log(np.maximum(stable, 1e-9)).mean()))
             if len(stable) else 1.0)
        tau = -1.0 / np.log(g) if g < 1.0 else np.inf
        tau_rank[j] = min(tau, JAC_TAU_CAP)
    return {"tau_rank": tau_rank.tolist(),
            "frac_unstable": frac_unstable.tolist(),
            "unstable_total": float(frac_unstable.mean())}


def _spearman(a, b):
    """Rank correlation of two 1-D arrays."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


def half_probes(arm, tau, device, out_root="results"):
    """Ridge-decode block pose / agent pos from slow vs fast dim halves.

    Also reports the continuous trend: does the per-dim correlation with
    block/agent rise/fall with the dim's tau? A flat trend means the
    information is NOT organized along coordinate time scales.
    """
    enc, _, _ = load_policy(arm, out_root, device)
    dnp = np.load(PROBE_NPZ)
    n = 4000
    Z = encode_batch(enc, dnp["imgs"][:n], device)
    order = np.argsort(tau)[::-1]
    half = len(order) // 2
    out = {}
    for name, idx in [("slow", order[:half]), ("fast", order[half:]),
                      ("full", np.arange(len(tau)))]:
        out[name] = {
            "block": float(ridge_r2(Z[:, idx], dnp["block"][:n]).mean()),
            "agent": float(ridge_r2(Z[:, idx], dnp["agent"][:n]).mean()),
        }
    Zc = Z - Z.mean(axis=0)

    def dimcorr(Y):
        Yc = Y - Y.mean(axis=0)
        sd = Zc.std(axis=0)[:, None] * Yc.std(axis=0)[None, :]
        return np.abs(Zc.T @ Yc / (n * np.maximum(sd, 1e-8))).max(axis=1)

    out["trend"] = {
        "tau_vs_blockcorr": _spearman(tau, dimcorr(dnp["block"][:n])),
        "tau_vs_agentcorr": _spearman(tau, dimcorr(dnp["agent"][:n])),
    }
    return out


def eigenmode_probes(arm, obs, actions, starts, ends, enc, device,
                     n_points=128, seed=11, out_root="results"):
    """Probe block/agent from slow vs fast eigenmodes of the mean Jacobian.

    The encoder's coordinates homogenize per-dim tau (dense mixing), so
    this asks whether time-scale structure is organized in the dynamical
    eigenbasis instead - which is invariant to coordinate mixing.
    """
    ckpt = torch.load(os.path.join(out_root, arm, "world_model.pt"),
                      map_location="cpu")
    pred = Predictor()
    pred.load_state_dict(ckpt["pred"])
    pred.eval()
    rng = np.random.default_rng(seed)
    val = [i for i in range(len(starts)) if i % 10 == 9]
    frames, acts = [], []
    for _ in range(n_points):
        e = val[int(rng.integers(len(val)))]
        s, en = int(starts[e]), int(ends[e])
        t = int(rng.integers(s, en - 1))
        frames.append(episode_stacks(obs, s, en)[t - s])
        acts.append(actions[t] / 512.0)
    zs = torch.tensor(encode_batch(enc, np.stack(frames), device))
    aa = torch.tensor(np.stack(acts), dtype=torch.float32)

    def f(z, a):
        return pred(z.unsqueeze(0), a.unsqueeze(0)).squeeze(0)

    J = vmap(jacrev(f, argnums=0))(zs, aa).detach().numpy()
    lam, V = np.linalg.eig(J.mean(axis=0))
    order = np.argsort(-np.abs(lam))
    lam, V = lam[order], V[:, order]

    dnp = np.load(PROBE_NPZ)
    n = 4000
    Z = encode_batch(enc, dnp["imgs"][:n], device)
    half = len(lam) // 2
    out = {"lambda_split": [float(np.abs(lam[half - 1])),
                            float(np.abs(lam[half]))],
           "unstable_slow": int((np.abs(lam[:half]) >= 1.0).sum())}
    for name, cols in [("slow", slice(0, half)),
                       ("fast", slice(half, None))]:
        D = np.concatenate([V[:, cols].real.T, V[:, cols].imag.T], axis=0)
        F = Z @ D.T
        F = (F - F.mean(axis=0)) / np.maximum(F.std(axis=0), 1e-8)
        out[name] = {
            "block": float(ridge_r2(F, dnp["block"][:n]).mean()),
            "agent": float(ridge_r2(F, dnp["agent"][:n]).mean()),
        }
    return out


def band_metrics(arm, tau, z_eps, device, out_root="results"):
    """Per-band audit for the tv arm, scoring the PREREGISTERED criteria.

    Operationalization (fixed before training):
      c1  inter-band gap: max adjacent ratio of band-median tau >= 1.5 with
          band medians monotonically decreasing (structure in the intended
          direction, not a random inversion);
      c2  block probe: slow band (0) minus fast band (K-1) >= 0.2;
      c3  agent probe: fast band (K-1) >= 0.3.
    Bands are the contiguous training blocks (band 0 = highest T_k, the slow
    target). alive_bands guards against pseudo-slow collapse: mean band std
    must exceed 0.01 (the collapsed energy-only control measured 0.003).
    """
    enc, _, _ = load_policy(arm, out_root, device)
    dnp = np.load(PROBE_NPZ)
    n = 4000
    Z = encode_batch(enc, dnp["imgs"][:n], device)
    w = Z.shape[1] // N_BANDS
    std_dim = np.concatenate(z_eps).std(axis=0)
    out = {"band_mult": BAND_MULT, "band_med_tau": [], "band_mean_std": [],
           "probes": {}}
    for k in range(N_BANDS):
        idx = np.arange(k * w, (k + 1) * w)
        out["band_med_tau"].append(float(np.median(tau[idx])))
        out["band_mean_std"].append(float(std_dim[idx].mean()))
        out["probes"][str(k)] = {
            "block": float(ridge_r2(Z[:, idx], dnp["block"][:n]).mean()),
            "agent": float(ridge_r2(Z[:, idx], dnp["agent"][:n]).mean()),
        }
    meds = out["band_med_tau"]
    out["band_ratios"] = [float(meds[k] / max(meds[k + 1], 1e-9))
                          for k in range(N_BANDS - 1)]
    mono = all(meds[i] >= meds[i + 1] for i in range(N_BANDS - 1))
    out["criteria"] = {
        "c1_interband_gap": bool(max(out["band_ratios"]) >= 1.5 and mono),
        "c2_block_slow_minus_fast": bool(
            out["probes"]["0"]["block"]
            - out["probes"][str(N_BANDS - 1)]["block"] >= 0.2),
        "c3_agent_fast": bool(
            out["probes"][str(N_BANDS - 1)]["agent"] >= 0.3),
    }
    out["alive_bands"] = bool(min(out["band_mean_std"]) >= 0.01)
    return out


def plot(results, out_root="results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"bc": "pixel BC", "bc_flow": "BC+flow", "jepa": "JEPA+VICReg",
             "e2jepa": "E2-JEPA", "energy_only": "energy only (T=0)",
             "tv": "E2-JEPA + T_k"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    ax = axes[0, 0]
    for arm, entry in results.items():
        ts = np.asarray(entry["tau_sorted"])
        ax.plot(np.arange(1, len(ts) + 1), ts, label=names[arm])
    ax.set_yscale("log")
    gap = results.get("e2jepa", {}).get("tau_gap")
    if gap:
        ax.axvline(gap["after_rank"], color="gray", ls=":", lw=1)
        ax.annotate("largest gap", (gap["after_rank"], gap["tau_above"]),
                    fontsize=8, color="gray",
                    xytext=(gap["after_rank"] + 4, gap["tau_above"] * 1.5))
    ax.set_xlabel("latent dim (sorted by tau)")
    ax.set_ylabel("tau (env steps)")
    ax.set_title("A. data-side tau spectrum (autocorrelation)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for arm, entry in results.items():
        if "jac" not in entry:
            continue
        tb = np.asarray(entry["jac"]["tau_rank"])
        ax.plot(np.arange(1, len(tb) + 1), tb, "--",
                label=f"{names[arm]}: model (eigen)")
        ts = np.asarray(entry["tau_sorted"])
        ax.plot(np.arange(1, len(ts) + 1), ts, "-", alpha=0.6,
                label=f"{names[arm]}: data (ACF)")
    ax.set_yscale("log")
    ax.set_xlabel("mode / dim rank")
    ax.set_ylabel("tau (env steps)")
    ax.set_title("B. model-side (Jacobian) vs data-side")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    arms = list(results)
    xs = np.arange(len(arms))
    w = 0.2
    bars = [("slow", "block", "#02743B"), ("fast", "block", "#7FD6A8"),
            ("slow", "agent", "#1F5FA8"), ("fast", "agent", "#80C1FF")]
    for i, (key, target, color) in enumerate(bars):
        vals = [results[a]["probes"][key][target] for a in arms]
        ax.bar(xs + (i - 1.5) * w, vals, w, color=color,
               label=f"{key} dims -> {target}")
    ax.set_xticks(xs, [names[a] for a in arms], fontsize=8)
    ax.set_ylabel("linear probe R^2")
    ax.set_title("slow vs fast halves decode different things")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ears = [a for a in results if "eigen_probes" in results[a]]
    xs = np.arange(len(ears))
    for i, (key, target, color) in enumerate(bars):
        vals = [results[a]["eigen_probes"][key][target] for a in ears]
        ax.bar(xs + (i - 1.5) * w, vals, w, color=color,
               label=f"{key} modes -> {target}")
    ax.set_xticks(xs, [names[a] for a in ears], fontsize=8)
    ax.set_ylabel("linear probe R^2")
    ax.set_title("eigenmodes of the learned dynamics (mean J)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "tau_spectrum.png"), dpi=140)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--dmax", type=int, default=DMAX)
    p.add_argument("--out_root", default="results")
    args = p.parse_args()

    device = device_auto()
    obs, actions, _, starts, ends, _ = load_cache()
    results = {}
    for arm in args.arms:
        print(f"[tau:{arm}] encoding episodes ...", flush=True)
        enc, _, _ = load_policy(arm, args.out_root, device)
        z_eps = encode_episodes(enc, obs, starts, ends, device)
        tau, censored = tau_acf(z_eps, args.dmax)
        ts = np.sort(tau)[::-1]
        entry = {
            "tau_sorted": ts.tolist(),
            "tau_gap": largest_gap(ts),
            "tau_quartiles": [float(np.quantile(tau, q))
                              for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
            "censored_frac": float(censored.mean()),
            "participation_ratio": participation(z_eps),
            "mean_dim_std": float(np.concatenate(z_eps).std(axis=0).mean()),
            "probes": half_probes(arm, tau, device, args.out_root),
        }
        if os.path.exists(os.path.join(args.out_root, arm,
                                       "world_model.pt")):
            print(f"[tau:{arm}] jacobian spectrum ...", flush=True)
            entry["jac"] = tau_jacobian(arm, obs, actions, starts, ends,
                                        enc, device, out_root=args.out_root)
            print(f"[tau:{arm}] eigenmode probes ...", flush=True)
            entry["eigen_probes"] = eigenmode_probes(
                arm, obs, actions, starts, ends, enc, device,
                out_root=args.out_root)
        results[arm] = entry
        if arm == "tv":
            print(f"[tau:{arm}] band audit (preregistered criteria) ...",
                  flush=True)
            entry["bands"] = band_metrics(arm, tau, z_eps, device,
                                          args.out_root)
            b = entry["bands"]
            print(f"[tau:{arm}] BANDS med_tau="
                  f"{[round(v, 1) for v in b['band_med_tau']]} "
                  f"ratios={[round(r, 2) for r in b['band_ratios']]} "
                  f"std={[round(v, 3) for v in b['band_mean_std']]} "
                  f"criteria={b['criteria']} alive={b['alive_bands']}",
                  flush=True)
        pr = entry["probes"]
        gap = entry["tau_gap"]
        med = entry["tau_quartiles"][2]
        tr = pr["trend"]
        print(f"[tau:{arm}] PR={entry['participation_ratio']:.1f} "
              f"median tau={med:.1f} "
              f"gap x{gap['ratio']:.2f}@{gap['after_rank']} | "
              f"block s/f={pr['slow']['block']:.2f}/{pr['fast']['block']:.2f} "
              f"agent s/f={pr['slow']['agent']:.2f}/"
              f"{pr['fast']['agent']:.2f} | "
              f"trend blk={tr['tau_vs_blockcorr']:+.2f} "
              f"ag={tr['tau_vs_agentcorr']:+.2f}", flush=True)
        if "eigen_probes" in entry:
            ep = entry["eigen_probes"]
            print(f"[tau:{arm}] eigenmodes block s/f="
                  f"{ep['slow']['block']:.2f}/{ep['fast']['block']:.2f} "
                  f"agent s/f={ep['slow']['agent']:.2f}/"
                  f"{ep['fast']['agent']:.2f}", flush=True)
    out = os.path.join(args.out_root, "tau_spectrum.json")
    if os.path.exists(out):  # merge: never clobber other arms' entries
        merged = json.load(open(out))
        merged.update(results)
        results = merged
    json.dump(results, open(out, "w"), indent=2)
    plot(results, args.out_root)
    print(f"wrote {out} and {args.out_root}/tau_spectrum.png")


if __name__ == "__main__":
    main()
