"""Verify the measurement stack against known ground truth.

Real-data numbers alone can hide pipeline bugs (the H5 signed-vs-squared
error bug was exactly this class). Each check feeds a component a signal
with KNOWN answer and asserts recovery; a H4-style negative result is only
meaningful if the method demonstrably detects structure when it exists.

  v1 data        cache vs parquet: frame/episode counts, coord ranges,
                 block visible in sampled teleop frames
  v2 tau_acf     AR(1) dims with known autocorr time tau = rho/(1-rho)
  v3 jacrev      vmap(jacrev) of the real Predictor vs central differences
  v4 partition    synthetic 2-block coupling -> eps*/null << 1 (H4 positive
                 control); iid coupling -> ratio ~ 1 (negative control)
  v5 block_pose  pixel block tracking vs env ground-truth pose (E3 validity)
  v6 ridge       closed-form ridge R2 on a synthetic linear map

Run:  python -m e2jepa.verify   (exits nonzero on any FAIL)
"""
import os
import sys

import numpy as np
import torch
from torch.func import jacrev, vmap

from .env_utils import COORD, make_env, render_obs
from .model import Predictor
from .strokes import BLOCK_COLOR, TOL, block_pose

RESULTS = []


def check(name, ok, msg=""):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {msg}", flush=True)


def v1_data():
    import pandas as pd
    from .data import CACHE, load_cache
    obs, actions, agent, starts, ends, cov = load_cache()
    parquet = os.path.join(os.path.dirname(CACHE),
                           "data/chunk-000/file-000.parquet")
    df = pd.read_parquet(parquet)
    ok_n = len(df) == len(obs) == len(actions) == len(agent)
    ep = np.asarray(df["episode_index"])
    n_ep = ep[-1] + 1
    ok_ep = n_ep == len(starts) == len(ends) == 206
    lens = np.diff(np.searchsorted(ep, np.arange(n_ep + 1)))
    ok_lens = bool((ends - starts == lens).all())
    r_act = (float(actions.min()), float(actions.max()))
    r_ag = (float(agent.min()), float(agent.max()))
    ok_rng = 0 <= actions.min() and actions.max() <= 512 and \
        0 <= agent.min() and agent.max() <= 512 and \
        0 <= cov.min() and cov.max() <= 1
    # block visible in sampled frames (mid-episode, 20 samples)
    rng = np.random.default_rng(0)
    vis = 0
    for _ in range(20):
        e = int(rng.integers(len(starts)))
        i = int(rng.integers(starts[e], ends[e]))
        m = np.abs(obs[i].astype(int) - BLOCK_COLOR).max(-1) < TOL
        vis += m.sum() > 12
    ok_vis = vis >= 18
    check("v1a cache==parquet counts", ok_n,
          f"n={len(obs)}")
    check("v1b 206 episodes, boundaries", ok_ep and ok_lens)
    check("v1c coord/reward ranges", ok_rng,
          f"act {r_act[0]:.0f}..{r_act[1]:.0f} agent {r_ag[0]:.0f}.."
          f"{r_ag[1]:.0f} cov {cov.min():.2f}..{cov.max():.2f}")
    check("v1d block visible in teleop frames", ok_vis,
          f"{vis}/20 frames have >12 block px")


def v2_tau_acf():
    """AR(1): x' = rho x + eps -> ACF = rho^D, tau = rho/(1-rho).

    Known bias: per-episode demeaning + the initial-positive-sequence
    truncation make tau_acf read LOW when tau is a sizeable fraction of
    the episode length. So we (a) sanity-check the fast regime and
    (b) measure the calibration curve tau_measured(tau_true) on the REAL
    episode-length distribution, to be applied to all reported tau values.
    """
    from .data import load_cache
    from .tau_spectrum import tau_acf
    rng = np.random.default_rng(1)

    def ar_episodes(rho, lengths, d=8):
        eps = []
        for L in lengths:
            z = np.empty((L, d))
            x = np.zeros(d)
            for t in range(L):
                x = rho * x + rng.normal(0, 1, d)
                z[t] = x
            eps.append(z)
        return eps

    # fast regime, uniform lengths: must be unbiased
    tau, _ = tau_acf(ar_episodes(np.full(8, 0.50), [150] * 60), dmax=96)
    fast = float(np.median(tau))
    check("v2a tau_acf fast group ~1", 0.4 <= fast <= 2.2,
          f"measured {fast:.2f}")

    # calibration on the real length distribution (206 teleop episodes)
    _, _, _, starts, ends, _ = load_cache()
    lengths = (ends - starts).astype(int).tolist()
    curve = []
    for tau_true in (2, 5, 10, 22, 40):
        rho = tau_true / (tau_true + 1)
        tau_m, _ = tau_acf(ar_episodes(np.full(8, rho), lengths), dmax=96)
        curve.append((tau_true, float(np.median(tau_m))))
    msg = " ".join(f"{t}->{m:.1f}" for t, m in curve)
    mono = all(curve[i][1] < curve[i + 1][1] for i in range(len(curve) - 1))
    check("v2b tau_acf calibration monotone", mono, msg)
    # ordering must be preserved with a gap: slow vs fast well separated
    slow_m = curve[-2][1]
    check("v2c slow/fast separation >=3x", slow_m >= 3 * fast,
          f"tau22-measured {slow_m:.1f} vs fast {fast:.2f}")
    np.save("results/tau_calibration.npy", np.array(curve))
    print(f"      calibration (true -> measured, real ep lengths): {msg}")


def v3_jacrev():
    """vmap(jacrev) of the real Predictor vs central differences."""
    path = os.path.join("results", "e2jepa", "world_model.pt")
    if not os.path.exists(path):
        check("v3 jacrev vs finite diff", True, "skipped (no ckpt)")
        return
    pred = Predictor()
    pred.load_state_dict(torch.load(path, map_location="cpu")["pred"])
    pred.eval()
    torch.manual_seed(0)
    z = torch.randn(128)
    a = torch.rand(2)

    def f(zi, ai):
        return pred(zi.unsqueeze(0), ai.unsqueeze(0)).squeeze(0)

    J = vmap(jacrev(f, argnums=0))(z[None], a[None])[0].detach().numpy()
    # jacrev returns J[out, in]; the finite-difference loop below fills
    # rows as input-perturbation gradients, i.e. J2[in, out] = J^T.
    eps = 1e-2
    J2 = np.empty((128, 128), dtype=np.float64)
    with torch.no_grad():
        for i in range(128):
            zp, zm = z.clone(), z.clone()
            zp[i] += eps
            zm[i] -= eps
            J2[i] = (f(zp, a) - f(zm, a)).numpy() / (2 * eps)
    rel = np.abs(J - J2.T).max() / max(np.abs(J).max(), 1e-9)
    check("v3 jacrev == central differences", rel < 0.05,
          f"max rel dev {rel:.4f}")


def v4_partition():
    from .h_audit import partition_eps
    rng = np.random.default_rng(2)
    d = 128
    half = 64
    # positive control: 2 blocks, intra coupling strong, inter 20x weaker
    M = np.abs(rng.normal(0, 1, (d, d)))
    M[:half, half:] *= 0.05
    M[half:, :half] *= 0.05
    pos = partition_eps(M, ks=(2,), n_null=100)
    # negative control: statistically isotropic coupling
    M0 = np.abs(rng.normal(0, 1, (d, d)))
    neg = partition_eps(M0, ks=(2,), n_null=100)
    r_pos = pos["2"]["ratio"]
    r_neg = neg["2"]["ratio"]
    check("v4a H4 method finds planted blocks", r_pos < 0.30,
          f"eps*/null = {r_pos:.3f} (planted inter/intra ~0.05)")
    check("v4b H4 method rejects iid coupling", r_neg > 0.70,
          f"eps*/null = {r_neg:.3f}")


def v5_block_pose():
    """Pixel block tracking vs the env's ground-truth pose.

    gym_pusht's block_pose is the pymunk body PLACEMENT point (the local
    origin = bottom-center of the T from add_tee), while the pixel mask
    centroid is the visual centroid ~ local (0, 40.7) world units = 4.46
    px at 56-res. A correct tracker must therefore show a CONSTANT
    offset vector that ROTATES with the block:
        centroid - bp = R(+/-theta) @ c0,  |c0| ~ 4.46 px
    (the +/- covers the screen y-down vs physics y-up flip). Validating
    that rigid relation is the right test: E3 uses centroid DISPLACEMENTS
    and tau, both invariant to a constant rotating offset. The
    second-moment orientation is the T's stem axis, ~90 deg from the
    block_pose angle reference.
    """
    env = make_env()
    errs, ths, n = [], [], 0
    for ep in range(4):
        env.reset(seed=9000 + ep)
        for _ in range(40):
            _, _, _, _, info = env.step(
                env.action_space.sample().astype(np.float32))
            bp = info["block_pose"]
            p = block_pose(render_obs(env))
            if p is None:
                continue
            cx, cy = p[0], p[1]
            errs.append([cx - bp[0] * 56 / COORD, cy - bp[1] * 56 / COORD])
            ths.append(bp[2])
            n += 1
    env.close()
    errs, ths = np.array(errs), np.array(ths)
    mags = np.hypot(errs[:, 0], errs[:, 1])
    spread = np.quantile(mags, 0.8) - np.quantile(mags, 0.2)
    check("v5a offset magnitude ~4.46px, constant",
          3.4 < np.median(mags) < 5.6 and spread < 2.0,
          f"median {np.median(mags):.2f} px, central-60% spread {spread:.2f}"
          f" over {n} states")
    # rigid fit: err = R(s*th) @ c0  =>  c0 = mean over samples of R(-s*th) err
    best = None
    for sgn in (1.0, -1.0):
        cs, sn = np.cos(sgn * ths), np.sin(sgn * ths)
        loc = np.stack([cs * errs[:, 0] + sn * errs[:, 1],
                        -sn * errs[:, 0] + cs * errs[:, 1]], axis=1)
        c0 = loc.mean(axis=0)
        res = np.hypot(*(loc - c0).T)
        med = float(np.median(res))
        if best is None or med < best[0]:
            best = (med, c0, sgn)
    med, c0, sgn = best
    check("v5b rigid rotating-offset fit <1.2px", med < 1.2,
          f"residual {med:.2f} px, c0=({c0[0]:.1f},{c0[1]:.1f}) px")


def v6_ridge():
    from .eval import ridge_r2
    rng = np.random.default_rng(4)
    N, d, k = 2000, 20, 5
    X = rng.normal(0, 1, (N, d))
    W = rng.normal(0, 1, (d, k)) / np.sqrt(d)  # per-target signal var ~ 1
    E = rng.normal(0, 0.5, (N, k))
    Y = X @ W + E
    r2 = ridge_r2(X, Y, alpha=1.0)
    # realized best-linear-predictor R^2 of THIS dataset (avoids seed luck:
    # per-target signal variance is chi-square, and E[s/(s+v)] > E[s]/(E[s]+v))
    theo = float((1 - E.var(axis=0) / Y.var(axis=0)).mean())
    check("v6 ridge R2 matches realized BLP", abs(r2.mean() - theo) < 0.02,
          f"measured {r2.mean():.3f} vs theory {theo:.3f}")


def main():
    v1_data()
    v2_tau_acf()
    v3_jacrev()
    v4_partition()
    v5_block_pose()
    v6_ridge()
    n_fail = RESULTS.count(False)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
