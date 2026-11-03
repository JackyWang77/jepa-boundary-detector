"""E4: H-standard audit of existing world models (zero training).

The axioms say a *usable* hierarchy must live in the code's own coordinates:
  H1  stratification - the tau spectrum has width (measured, tau_spectrum)
  H2  condensation   - slow dims carry slow content (block), fast carry agent
  H3  concentration - a SMALL slow subset is sufficient: block pose decodes
                       from top-k tau dims near full accuracy at small k,
                       and that subset is dynamically closed
  H4  Simon epsilon  - the learned dynamics is near-decomposable: some
                       partition of coordinates makes inter-block coupling
                       much weaker than intra-block coupling
  H5  assemblability - the slow block can be iterated to the task's own slow
                       scale (dwell median 16, tau_block ~22 steps) without
                       dying; the assembly law needs composable blocks

Preregistered pass rules (fixed before looking at any number):
  alive  mean_dim_std >= 0.01 (else the audit self-exposes as invalid;
         the collapsed energy_only control measured 0.003)
  H3a    R2(block | top-8 tau dims) >= 0.8 * R2(block | all 128 dims)
  H3b    R2(z_slow(t+h) | z_slow(t)) >= 0.9 * R2(z_slow(t+h) | z(t))
         for every h in {1, 4, 16}, slow = top-32 tau dims
  H4     exists K in {2..5}: eps*(K) <= 0.5 * eps_null(K), where eps_null is
         the size-matched random-partition null
  H5     full-rollout slow-content error at h=16 <= 0.5 AND slow-projected
         rollout error at h=16 <= 0.5

Prediction from the axioms: e2jepa / tv / jepa pass H1 but fail H3a / H4 /
H5 - LayerNorm dense mixing destroys coordinate blocks, so no small subset
concentrates the slow content and no projected block survives iteration of
the learned map. energy_only fails `alive`. If any self-organized arm passes
a criterion, the theory is wrong there and we learn that instead.

Everything here is zero-training: existing ckpts + ridge fits + rollouts.
H3 is measured under the naturalistic teleop distribution (no action
features): both variants of each comparison share the same conditioning,
so gaps are attributable to the z-features alone.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.func import jacrev, vmap

from .data import load_cache
from .eval import PROBE_NPZ, encode_batch, ridge_r2
from .model import Encoder, Predictor
from .tau_spectrum import encode_episodes, episode_stacks, tau_acf
from .train import device_auto

JAC_POINTS = 240     # trajectory points for the Jacobian coupling matrix
HORIZONS = [1, 4, 16]
K_BLOCKS = [2, 3, 4, 5]
N_NULL = 200         # random size-matched partitions per K
SLOW_DIMS = 32
CONC_KS = [4, 8, 16, 32, 64, 128]
H5_H = 32            # rollout horizon (env steps)
H5_WINDOWS = 300
DIRECT_HS = [1, 2, 4, 8, 16, 32]
ALIVE_STD = 0.01


def load_world_model(arm, out_root="results", device=None):
    device = device or device_auto()
    ckpt = torch.load(os.path.join(out_root, arm, "world_model.pt"),
                      map_location="cpu")
    enc = Encoder().to(device)
    enc.load_state_dict(ckpt["enc"])
    enc.eval()
    return enc, ckpt


def ridge_split(Ztr, Ytr, Zte, Yte, alpha=1.0):
    """Ridge with an explicit train/test split (episodes never mix)."""
    Ztr = np.c_[Ztr, np.ones(len(Ztr))]
    Zte = np.c_[Zte, np.ones(len(Zte))]
    W = np.linalg.solve(Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1]),
                        Ztr.T @ Ytr)
    P = Zte @ W
    ss_res = ((Yte - P) ** 2).mean(axis=0)
    ss_tot = ((Yte - Yte.mean(axis=0)) ** 2).mean(axis=0)
    return 1 - ss_res / (ss_tot + 1e-12)


def concentration(enc, tau, device):
    """H3a: does a SMALL tau-ranked subset reach the full block decode?

    References: PCA-variance ranking and random dims. If tau-ranked ~
    random-ranked, tau carries no concentration information and there is
    no privileged slow subset (the redundancy regime of a flat code).
    """
    if not os.path.exists(PROBE_NPZ):
        from .env_utils import random_probe_set
        random_probe_set(out_path=PROBE_NPZ)
    dnp = np.load(PROBE_NPZ)
    n = 4000
    Z = encode_batch(enc, dnp["imgs"][:n], device)
    Y = dnp["block"][:n]
    d = Z.shape[1]
    order = np.argsort(tau)[::-1]
    Zc = Z - Z.mean(axis=0)
    _, V = np.linalg.eigh(Zc.T @ Zc / n)
    V = V[:, ::-1]
    rng = np.random.default_rng(0)
    out = {"k": CONC_KS, "r2_tau": [], "r2_pca": [], "r2_rand": []}
    for k in CONC_KS:
        out["r2_tau"].append(float(ridge_r2(Z[:, order[:k]], Y).mean()))
        out["r2_pca"].append(float(ridge_r2(Z @ V[:, :k], Y).mean()))
        out["r2_rand"].append(float(np.mean([
            ridge_r2(Z[:, rng.choice(d, k, replace=False)], Y).mean()
            for _ in range(3)])))
    out["r2_full"] = float(ridge_r2(Z, Y).mean())
    return out


def closure(z_eps, tau, k_slow=SLOW_DIMS, horizons=HORIZONS):
    """H3b: dynamical closure of the slow subset, model-free.

    If the slow block were closed, knowing only its own coordinates would
    predict its own future nearly as well as knowing the full latent.
    """
    order = np.argsort(tau)[::-1]
    S = order[:k_slow]
    tr = [z for e, z in enumerate(z_eps) if e % 10 != 9]
    te = [z for e, z in enumerate(z_eps) if e % 10 == 9]
    out = {"h": list(horizons), "slow_only": [], "full_feat": [], "gap": []}
    for h in horizons:
        def pairs(eps):
            X = np.concatenate([z[:-h] for z in eps if len(z) > h])
            Y = np.concatenate([z[h:] for z in eps if len(z) > h])
            return X, Y
        Xtr, Ytr = pairs(tr)
        Xte, Yte = pairs(te)
        r_so = float(ridge_split(Xtr[:, S], Ytr[:, S], Xte[:, S],
                                 Yte[:, S]).mean())
        r_fu = float(ridge_split(Xtr, Ytr[:, S], Xte, Yte[:, S]).mean())
        out["slow_only"].append(r_so)
        out["full_feat"].append(r_fu)
        out["gap"].append(r_fu - r_so)
    out["k_slow"] = k_slow
    return out


def jacobian_stats(arm, obs, actions, starts, ends, enc, device, seed=5,
                   n_points=JAC_POINTS, out_root="results"):
    """H4 ingredients: mean |J| coupling matrix + eigen-mixing of mean J."""
    pred = Predictor()
    pred.load_state_dict(torch.load(os.path.join(out_root, arm,
                                                 "world_model.pt"),
                                    map_location="cpu")["pred"])
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

    J = vmap(jacrev(f, argnums=0))(zs, aa).detach().numpy()  # (n,d,d)
    M = np.abs(J).mean(axis=0)
    Jm = J.mean(axis=0)
    lam, V = np.linalg.eig(Jm)
    ipr = (np.abs(V) ** 4).sum(axis=0)
    d = M.shape[0]
    return M, {
        "eig_ipr_mean": float(ipr.mean()),
        "eig_ipr_uniform": 1.0 / d,
        "frac_localized_modes": float((ipr >= 4.0 / d).mean()),
        "jac_mean_abs_eigs": np.sort(np.abs(lam))[::-1].tolist(),
    }


def spectral_partition(S, K, restarts=12, seed=0):
    """Normalized-spectral clustering on the symmetrized coupling graph."""
    d = S.shape[0]
    dd = np.sqrt(np.maximum(S.sum(axis=1), 1e-12))
    L = np.eye(d) - S / dd[:, None] / dd[None, :]
    _, v = np.linalg.eigh(L)
    X = v[:, :K]
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    best_lab, best_cost = None, np.inf
    for _ in range(restarts):
        mu = X[rng.integers(d, size=K)].copy()
        lab = np.zeros(d, dtype=int)
        for _ in range(200):
            lab = ((X[:, None, :] - mu[None]) ** 2).sum(-1).argmin(axis=1)
            for k in range(K):
                m = lab == k
                if m.any():
                    mu[k] = X[m].mean(axis=0)
        cost = float(((X - mu[lab]) ** 2).sum())
        if cost < best_cost:
            best_cost, best_lab = cost, lab.copy()
    return best_lab


def eps_of(M, lab):
    """Simon-epsilon: mean inter-block |coupling| / mean intra-block."""
    same = lab[:, None] == lab[None, :]
    offd = ~np.eye(len(M), dtype=bool)
    A = np.abs(M)
    intra = A[same & offd].mean() if (same & offd).any() else 1.0
    inter = A[~same].mean() if (~same).any() else 1.0
    return float(inter / max(intra, 1e-12))


def partition_eps(M, ks=K_BLOCKS, n_null=N_NULL, seed=0):
    """H4: best achievable near-decomposability vs size-matched null."""
    S = M + M.T
    rng = np.random.default_rng(seed)
    out = {}
    for K in ks:
        lab = spectral_partition(S, K, seed=seed + K)
        eps = eps_of(M, lab)
        null = np.array([eps_of(M, rng.permutation(lab))
                         for _ in range(n_null)])
        out[str(K)] = {
            "epsilon": eps,
            "null_mean": float(null.mean()),
            "null_std": float(null.std()),
            "ratio": float(eps / null.mean()),
            "sizes": np.bincount(lab, minlength=K).tolist(),
        }
    return out


def rollout_curves(arm, z_eps, actions, starts, tau, device,
                   seed=7, h_max=H5_H, n_windows=H5_WINDOWS,
                   k_slow=SLOW_DIMS, out_root="results"):
    """H5: three ways to know the slow content at horizon h.

      full  - iterate the trained one-step model (the planner's micro map)
      proj  - same, but after every step discard the fast coordinates:
              assembling a "slow block" out of a flat code
      direct- ridge z_t -> z_{t+h} fit per horizon (the macro oracle)

    Errors are per-dim MSE / train-variance, averaged over the slow dims;
    1.0 = uninformed (predicting the mean). h* = where the full rollout
    crosses 0.9.
    """
    ckpt = torch.load(os.path.join(out_root, arm, "world_model.pt"),
                      map_location="cpu")
    pred = Predictor().to(device)
    pred.load_state_dict(ckpt["pred"])
    pred.eval()
    order = np.argsort(tau)[::-1]
    slow = order[:k_slow]
    tr_eps = [z for e, z in enumerate(z_eps) if e % 10 != 9]
    var = np.concatenate(tr_eps).var(axis=0)

    val = [e for e in range(len(starts)) if e % 10 == 9]
    rng = np.random.default_rng(seed)
    W, tries = [], 0
    while len(W) < n_windows and tries < 5000:
        tries += 1
        e = val[int(rng.integers(len(val)))]
        L = len(z_eps[e])
        if L <= h_max + 1:
            continue
        t0 = int(rng.integers(0, L - h_max - 1))
        W.append((e, t0))
    Z0 = np.stack([z_eps[e][t0] for e, t0 in W])
    GT = np.stack([z_eps[e][t0:t0 + h_max + 1] for e, t0 in W])  # (B,h+1,d)
    A = np.stack([actions[int(starts[e]) + t0:
                          int(starts[e]) + t0 + h_max] for e, t0 in W]) / 512.0

    z0t = torch.tensor(Z0, dtype=torch.float32, device=device)
    at = torch.tensor(A, dtype=torch.float32, device=device)
    mask = np.zeros(Z0.shape[1], dtype=np.float32)
    mask[slow] = 1.0
    mt = torch.tensor(mask, device=device)

    GTt = torch.tensor(GT, dtype=torch.float32, device=device)

    def run(project):
        errs, zi = [], z0t
        with torch.no_grad():
            for j in range(h_max):
                zi = pred(zi, at[:, j])
                if project:
                    zi = zi * mt
                d = (zi - GTt[:, j + 1]).cpu().numpy().astype(np.float64)
                errs.append(d * d)  # square in float64: no overflow
        return np.stack(errs)  # (h, B, d)

    def norm_err(E):
        per_dim = E.mean(axis=1) / np.maximum(var, 1e-12)  # (h, d)
        return per_dim[:, slow].mean(axis=1), per_dim.mean(axis=1)

    e_full = norm_err(run(False))
    e_proj = norm_err(run(True))

    direct = {"slow": {}, "all": {}}
    Z0b = np.c_[Z0, np.ones(len(Z0))]
    for h in DIRECT_HS:
        Xtr = np.concatenate([z[:-h] for z in tr_eps if len(z) > h])
        Ytr = np.concatenate([z[h:] for z in tr_eps if len(z) > h])
        Wf = np.linalg.solve(
            np.c_[Xtr, np.ones(len(Xtr))].T @ np.c_[Xtr, np.ones(len(Xtr))]
            + np.eye(Xtr.shape[1] + 1), np.c_[Xtr, np.ones(len(Xtr))].T @ Ytr)
        P = Z0b @ Wf
        per_dim = ((P - GT[:, h]) ** 2).mean(axis=0) / np.maximum(var, 1e-12)
        direct["slow"][str(h)] = float(per_dim[slow].mean())
        direct["all"][str(h)] = float(per_dim.mean())

    cross = np.nonzero(e_full[0] >= 0.9)[0]
    return {
        "h": list(range(1, h_max + 1)),
        "err_full_slow": e_full[0].tolist(),
        "err_full_all": e_full[1].tolist(),
        "err_proj_slow": e_proj[0].tolist(),
        "err_direct_slow": direct["slow"],
        "err_direct_all": direct["all"],
        "k_slow": k_slow,
        "n_windows": len(W),
        "h_star": int(cross[0] + 1) if len(cross) else None,
    }


def plot(results, out_root="results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    for arm in arms:
        c = results[arm]["h3"]["concentration"]
        ax.plot(c["k"], c["r2_tau"], marker="o", label=arm)
    if arms:
        c = results[arms[0]]["h3"]["concentration"]
        ax.plot(c["k"], c["r2_rand"], "--", color="gray", alpha=0.7,
                label=f"{arms[0]}: random-k")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("subset size k (tau-ranked dims)")
    ax.set_ylabel("block-pose probe R^2")
    ax.set_title("H3a: slow-content concentration")
    ax.legend(fontsize=7)

    ax = axes[1]
    xs = np.arange(len(K_BLOCKS))
    w = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        rs = [results[arm]["h4"]["partitions"][str(K)]["ratio"]
              for K in K_BLOCKS]
        ax.bar(xs + i * w, rs, w, label=arm)
    ax.axhline(0.5, color="gray", ls="--", lw=1, label="pass (<=0.5)")
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="null level")
    ax.set_xticks(xs + 0.4, [f"K={K}" for K in K_BLOCKS])
    ax.set_ylabel("epsilon* / epsilon_null")
    ax.set_title("H4: best partition vs random (Simon)")
    ax.legend(fontsize=7)

    ax = axes[2]
    alive_arms = [a for a in arms if results[a].get("alive", True)]
    for arm in alive_arms:
        r = results[arm]["h5"]
        ax.plot(r["h"], r["err_full_slow"], label=f"{arm}: rollout")
        ax.plot(r["h"], r["err_proj_slow"], ":", alpha=0.8,
                label=f"{arm}: slow-proj")
        ax.plot([int(h) for h in r["err_direct_slow"]],
                list(r["err_direct_slow"].values()), "--s", ms=4, alpha=0.8,
                label=f"{arm}: direct")
    if len(alive_arms) < len(arms):
        ax.annotate("collapsed arm(s) diverge (alive=False), not shown",
                    (1, 2.75), fontsize=7, color="gray")
    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.axvline(16, color="gray", ls="--", lw=1)
    ax.annotate("dwell median=16", (16, 1.05), fontsize=8, color="gray")
    ax.set_xlabel("horizon h (env steps)")
    ax.set_ylabel("slow-content error / var")
    ax.set_ylim(0, 3.0)
    ax.set_title("H5: error vs horizon (1.0 = uninformed)")
    ax.legend(fontsize=6, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "h_audit.png"), dpi=140)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+",
                   default=["e2jepa", "tv", "jepa", "energy_only"])
    p.add_argument("--out_root", default="results")
    args = p.parse_args()

    device = device_auto()
    obs, actions, _, starts, ends, _ = load_cache()
    results = {}
    for arm in args.arms:
        if not os.path.exists(os.path.join(args.out_root, arm,
                                           "world_model.pt")):
            print(f"[h:{arm}] no world_model.pt, skipping", flush=True)
            continue
        enc, _ = load_world_model(arm, args.out_root, device)
        z_eps = encode_episodes(enc, obs, starts, ends, device)
        dim_std = float(np.concatenate(z_eps).std(axis=0).mean())
        tau, _ = tau_acf(z_eps)
        entry = {"mean_dim_std": dim_std,
                 "tau_median": float(np.median(tau))}
        print(f"[h:{arm}] H3 concentration + closure ...", flush=True)
        entry["h3"] = {"concentration": concentration(enc, tau, device),
                       "closure": closure(z_eps, tau)}
        print(f"[h:{arm}] H4 jacobians + partitions ...", flush=True)
        M, jstats = jacobian_stats(arm, obs, actions, starts, ends, enc,
                                   device, out_root=args.out_root)
        entry["h4"] = jstats
        entry["h4"]["partitions"] = partition_eps(M)
        print(f"[h:{arm}] H5 rollout curves ...", flush=True)
        entry["h5"] = rollout_curves(arm, z_eps, actions, starts, tau,
                                     device, out_root=args.out_root)

        entry["alive"] = bool(dim_std >= ALIVE_STD)
        c = entry["h3"]["concentration"]
        k8 = c["r2_tau"][c["k"].index(8)]
        cl = entry["h3"]["closure"]
        parts = entry["h4"]["partitions"]
        r5 = entry["h5"]
        entry["criteria"] = {
            "h3a_concentration": bool(k8 >= 0.8 * c["r2_full"]),
            "h3b_closure": bool(all(
                so >= 0.9 * fu
                for so, fu in zip(cl["slow_only"], cl["full_feat"]))),
            "h4_simon_eps": bool(any(
                parts[str(K)]["epsilon"] <= 0.5 * parts[str(K)]["null_mean"]
                for K in K_BLOCKS)),
            "h5_assembly": bool(r5["err_full_slow"][15] <= 0.5
                                and r5["err_proj_slow"][15] <= 0.5),
        }
        best_k = min(K_BLOCKS, key=lambda K: parts[str(K)]["ratio"])
        print(f"[h:{arm}] alive={entry['alive']} std={dim_std:.3f} | "
              f"H3a top8 R2={k8:.2f} (full {c['r2_full']:.2f}, "
              f"rand8 {c['r2_rand'][1]:.2f}) "
              f"closure_gap_h16={cl['gap'][-1]:+.3f} | "
              f"H4 best eps/null="
              f"{parts[str(best_k)]['ratio']:.2f}@K{best_k} | "
              f"H5 err16 full={r5['err_full_slow'][15]:.2f} "
              f"proj={r5['err_proj_slow'][15]:.2f} "
              f"direct={r5['err_direct_slow']['16']:.2f} "
              f"h*={r5['h_star']}", flush=True)
        print(f"[h:{arm}] criteria={entry['criteria']}", flush=True)
        results[arm] = entry

    out = os.path.join(args.out_root, "h_audit.json")
    if os.path.exists(out):  # merge: never clobber other arms' entries
        merged = json.load(open(out))
        merged.update(results)
        results = merged
    json.dump(results, open(out, "w"), indent=2)
    plot(results, args.out_root)
    print(f"wrote {out} and {args.out_root}/h_audit.png")


if __name__ == "__main__":
    main()
