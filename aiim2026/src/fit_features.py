"""Fit SFA decomposition on train latents and extract three boundary features.

Features live on the latent increment stream dz_t = z_t - z_{t-1}:
  f1  fast-subspace energy ratio (window-5 smoothed, minus per-episode median)
  f2  cosine of consecutive slow-subspace increments
  f3  log increment energy
Kinematic reference events (push onset / settle) come from the block's own
motion, zero manual labels and independent of block shape/scale (the env's
blocks are polyominoes up to ~130px across, so agent-block distance cannot
define contact). Outputs: runs/features_{train,hold}.npz and
runs/feature_fit.json.
"""

import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parents[1] / 'runs'
K_SLOW = 16           # slow subspace dim (of 192)
BLOCK_DISP = 0.5      # block displacement threshold per step (px)
BLOCK_RUN = 2         # sustained steps to call push onset
SETTLE_DISP = 0.1     # displacement below which the block counts as parked
SETTLE_RUN = 3        # sustained quiet steps to call settle
SMOOTH_W = 5         # moving-average window for f1


def load(name):
    d = np.load(RUNS / f'latents_{name}.npz')
    return d['z'], d['state'], d['ep_len']


def iter_eps(z, state, ep_len):
    off = 0
    for n in ep_len:
        n = int(n)
        yield z[off:off + n], None if state is None else state[off:off + n]
        off += n


def fit_sfa(z_train, ep_len_train):
    """Closed-form linear SFA: whiten with signal covariance, then rotate by
    ascending eigenvalues of the whitened increment covariance."""
    mu = z_train.mean(axis=0)
    zc = z_train - mu
    cov = np.cov(zc, rowvar=False)
    lam, E = np.linalg.eigh(cov)          # ascending
    # guard against numerically zero eigenvalues
    lam = np.maximum(lam, 1e-8)
    Winv = E @ np.diag(lam ** -0.5)       # whiten: u = Winv.T @ (z - mu)

    incs = []
    for z_ep, _ in iter_eps(z_train, None, ep_len_train):
        incs.append(z_ep[1:] - z_ep[:-1])
    du = np.concatenate(incs, axis=0) @ Winv
    cov_du = np.cov(du, rowvar=False)
    theta, Q = np.linalg.eigh(cov_du)     # ascending = slowest first
    return {
        'mu': mu,
        'Winv': Winv,
        'theta': theta,
        'q_slow': Q[:, :K_SLOW],
        'q_fast': Q[:, -K_SLOW:],
    }


def kinematic_events(state):
    """Reference events from the block's own motion, independent of block
    shape/scale (distance-based contact is not definable here: polyomino
    blocks span up to ~130px, edge pushes happen at agent-block center
    distances of 50-90px, and blocks coast after release since the space
    has zero damping).
    push_onset: block displacement > BLOCK_DISP for >= BLOCK_RUN steps.
    settle:     after a motion run, displacement < SETTLE_DISP for
                >= SETTLE_RUN consecutive steps (first quiet frame).
    Event arrays are indexed by frame; caller aligns frame t -> feature
    index t-1."""
    T = len(state)
    disp = np.concatenate(
        [[0.0], np.linalg.norm(np.diff(state[:, 2:4], axis=0), axis=1)]
    )
    push_onset = np.zeros(T, bool)
    settle = np.zeros(T, bool)
    moving = disp > BLOCK_DISP
    t = 0
    while t < T:
        if moving[t]:
            j = t
            while j < T and moving[j]:
                j += 1
            if j - t >= BLOCK_RUN:
                push_onset[t] = True
                # settle: first frame of a >=SETTLE_RUN quiet stretch after j
                k, quiet, first_quiet = j, 0, None
                while k < T:
                    if disp[k] < SETTLE_DISP:
                        quiet += 1
                        if quiet == 1:
                            first_quiet = k
                        if quiet >= SETTLE_RUN:
                            settle[first_quiet] = True
                            break
                    else:
                        quiet = 0
                    k += 1
            t = j
        else:
            t += 1
    return push_onset, settle


# --- cube (OGB) event constants: calibrated 2026-11-05 on the 500 local
# oracle episodes (see runs/cube_event_thresh.json). Block jitter floor is
# < 1e-5 m/frame, deliberate motion is >= ~1e-3 m/frame; 5:1 onset:settle
# threshold ratio kept as in PushT.
CUBE_DISP = 5e-4      # block displacement threshold per step (m)
CUBE_RUN = 2          # sustained steps to call manip onset
CUBE_SETTLE_DISP = 1e-4   # displacement below which block counts as parked
CUBE_SETTLE_RUN = 3       # sustained quiet steps to call settle


def ogb_cube_events(block_pos):
    """Reference events from the cube's own motion (task-scoped twin of
    kinematic_events above; the PushT path is untouched).

    manip_onset: block displacement > CUBE_DISP for >= CUBE_RUN steps.
    settle:      after a motion run, displacement < CUBE_SETTLE_DISP for
                 >= CUBE_SETTLE_RUN consecutive steps (first quiet frame).
    Event arrays are indexed by frame; caller aligns frame t -> feature
    index t-1 (same convention as kinematic_events)."""
    T = len(block_pos)
    disp = np.concatenate(
        [[0.0], np.linalg.norm(np.diff(block_pos[:, :3], axis=0), axis=1)])
    manip_onset = np.zeros(T, bool)
    settle = np.zeros(T, bool)
    moving = disp > CUBE_DISP
    t = 0
    while t < T:
        if moving[t]:
            j = t
            while j < T and moving[j]:
                j += 1
            if j - t >= CUBE_RUN:
                manip_onset[t] = True
                k, quiet, first_quiet = j, 0, None
                while k < T:
                    if disp[k] < CUBE_SETTLE_DISP:
                        quiet += 1
                        if quiet == 1:
                            first_quiet = k
                        if quiet >= CUBE_SETTLE_RUN:
                            settle[first_quiet] = True
                            break
                    else:
                        quiet = 0
                    k += 1
            t = j
        else:
            t += 1
    return manip_onset, settle


def moving_avg(x, w):
    if len(x) < w:
        return x.copy()
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='same')


def extract_features(z_ep, fit):
    """Per-episode features. Feature index i (0-based) = frame i+1."""
    dz = z_ep[1:] - z_ep[:-1]                       # (T-1, 192)
    du = dz @ fit['Winv']
    slow = du @ fit['q_slow']
    fast = du @ fit['q_fast']

    e_total = np.sum(du * du, axis=1) + 1e-12
    f1_raw = np.sum(fast * fast, axis=1) / e_total  # in [0,1]
    f1 = moving_avg(f1_raw, SMOOTH_W)
    f1 = f1 - np.median(f1)                          # per-episode baseline

    n0 = np.linalg.norm(slow[:-1], axis=1)
    n1 = np.linalg.norm(slow[1:], axis=1)
    cos = np.sum(slow[1:] * slow[:-1], axis=1) / (n0 * n1 + 1e-12)
    f2 = np.concatenate([[cos[0] if len(cos) else 1.0], cos])

    f3 = np.log1p(np.sum(dz * dz, axis=1))
    return f1, f2, f3, f1_raw


def main():
    z_tr, s_tr, l_tr = load('train')
    z_ho, s_ho, l_ho = load('hold')
    fit = fit_sfa(z_tr, l_tr)

    for name, z, s, l in [('train', z_tr, s_tr, l_tr),
                          ('hold', z_ho, s_ho, l_ho)]:
        F1, F2, F3, F1R, P, S, L = [], [], [], [], [], [], []
        for z_ep, s_ep in iter_eps(z, s, l):
            f1, f2, f3, f1_raw = extract_features(z_ep, fit)
            push, settle = kinematic_events(s_ep)
            F1.append(f1); F2.append(f2); F3.append(f3); F1R.append(f1_raw)
            P.append(push[1:]); S.append(settle[1:])
            L.append(len(f1))
        np.savez_compressed(
            RUNS / f'features_{name}.npz',
            f1=np.concatenate(F1), f2=np.concatenate(F2),
            f3=np.concatenate(F3), f1_raw=np.concatenate(F1R),
            push_onset=np.concatenate(P), settle=np.concatenate(S),
            ep_len=np.array(L),
        )
        n_push = int(sum(p.sum() for p in P))
        n_settle = int(sum(s.sum() for s in S))
        print(f'[{name}] episodes={len(L)} points={int(sum(L))} '
              f'push_onsets={n_push} settles={n_settle}')

    fast_share = float(np.mean(np.load(
        RUNS / 'features_train.npz')['f1_raw']))
    fit_json = {
        'k_slow': K_SLOW,
        'block_disp_px': BLOCK_DISP,
        'settle_disp_px': SETTLE_DISP,
        'smooth_window': SMOOTH_W,
        'theta_min': float(fit['theta'][0]),
        'theta_max': float(fit['theta'][-1]),
        'fast_subspace_energy_share_train': fast_share,
    }
    with open(RUNS / 'feature_fit.json', 'w') as f:
        json.dump(fit_json, f, indent=2)
    print(json.dumps(fit_json, indent=2))


if __name__ == '__main__':
    main()
