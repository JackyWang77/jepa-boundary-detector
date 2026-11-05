"""Probe: does the R2 detector's p distribution shift on the model's own
generated rollouts vs real recorded latents?

For each episode: take the first 3 latent frames as context, auto-regressively
predict 15 blocks with the frozen LeWM using the real action blocks (so the
generated path diverges purely through model drift), then compute the exact
18-channel feature rows + CausalCNN head on BOTH the real sequence and the
generated sequence. Output: p stats per sequence, real vs generated.

This is the first diagnostic for the 'detector stops the rollout' idea —
it measures how out-of-distribution the deployed head is on its own model's
hallucinated latents, instead of arguing about it.
"""

import json
from pathlib import Path

import numpy as np
import torch

import stable_worldmodel as swm

from e4_train_head import CausalCNN

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / 'features'
RUNS = ROOT / 'runs'
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
CTX, H = 3, 15           # context frames / generated blocks
WINDOWS = (2, 4, 8)
N_FEAT = 3 * 2 * len(WINDOWS)   # 18


def load_model():
    model = swm.wm.utils.load_pretrained('pusht/weights.pt')
    model = model.to(DEVICE).eval()
    model.requires_grad_(False)
    return model


def load_head(meta_path):
    meta = json.load(open(meta_path))
    head = CausalCNN(in_ch=18)
    head.load_state_dict(torch.load(meta['head_ckpt'], map_location='cpu'))
    head.eval()
    head.requires_grad_(False)
    head.to(DEVICE)
    return head, meta


def make_rows(f1, f2, f3, mu, sd):
    """18-ch rows and per-step head probs for steps s=3.., mirroring the
    BoundaryPolicy._features row construction channel-for-channel."""
    T = len(f1)
    rows, valid_s = [], []
    for s in range(3, T):
        row = []
        for feat in (f1, f2, f3):
            for w in WINDOWS:
                hist = [v for v in feat[max(0, s - w + 1):s + 1]
                        if np.isfinite(v)]
                if not hist:
                    row.extend([0.0, 0.0])
                    continue
                arr = np.asarray(hist)
                row.append(float(arr.mean()))
                row.append(float(arr.std()) if len(arr) > 1 else 0.0)
        row = (np.asarray(row, np.float32) - mu) / sd
        rows.append(row)
        valid_s.append(s)
    return np.stack(rows), valid_s


def head_probs(head, rows):
    probs = []
    for i in range(len(rows)):
        x = torch.from_numpy(rows[:i + 1])[None].to(DEVICE)
        with torch.no_grad():
            logit = head(x)[0, -1]
        probs.append(float(torch.sigmoid(logit).item()))
    return probs


def feat_on_seq(z_seq, act_blocks, model, use_real_innov_target=True,
                fz=None):
    """f1/f2/f3 on a z sequence of length >= CTX+1.
    z_seq: list of (192,) numpy/tensor frames; act_blocks: (T,10) z-scored.
    fz: if given (action-blind linear map npz path), f3 = dual innov
        ||z[t+1] - W z[t-2..t]|| instead of the WM's own innovation.
    """
    z = z_seq if isinstance(z_seq, np.ndarray) else np.stack(z_seq)
    T = len(z)
    f1 = np.full(T, np.nan)
    f2 = np.full(T, np.nan)
    f3 = np.full(T, np.nan)
    dz = np.diff(z, axis=0)
    f1[1:] = np.linalg.norm(dz, axis=1)
    if T > 2:
        n0 = np.linalg.norm(dz[:-1], axis=1) + 1e-12
        n1 = np.linalg.norm(dz[1:], axis=1) + 1e-12
        f2[2:] = np.sum(dz[1:] * dz[:-1], axis=1) / (n0 * n1)
    if fz is not None:
        W = np.load(fz).astype(np.float32)
        zf = z.astype(np.float32)
        for t in range(3, T - 1):
            pred = zf[t - CTX:t].ravel() @ W
            f3[t] = float(np.linalg.norm(zf[t + 1] - pred))
        return f1, f2, f3
    for s in range(3, T):
        emb = torch.from_numpy(z[s - CTX:s]).float().to(DEVICE)[None]
        act = torch.from_numpy(act_blocks[s - CTX:s]).float().to(DEVICE)[None]
        with torch.no_grad():
            act_emb = model.action_encoder(act)
            pred = model.predict(emb, act_emb)
        tgt = z[s]          # real seq: real future frame; gen seq: generated
        f3[s] = float(np.linalg.norm(tgt - pred[0, -1].cpu().numpy()))
    return f1, f2, f3


def main():
    import sys
    dual = len(sys.argv) > 1 and sys.argv[1] == 'dual'
    model = load_model()
    meta_name = 'r2_meta_dual.json' if dual else 'r2_meta.json'
    out_name = 'probe_rollout_fire_dual.json' if dual else 'probe_rollout_fire.json'
    head, meta = load_head(ROOT / 'models' / meta_name)
    mu = np.asarray(meta['feat_mu'], np.float32)
    sd = np.asarray(meta['feat_sd'], np.float32)
    fz = str(ROOT / 'models' / 'fz_dual_linear.npy') if dual else None
    hold_ids = meta['hold_episode_ids']

    results = {}
    done = 0
    for eid in hold_ids:
        path = FEAT_DIR / f'ep{eid}.npz'
        if not path.exists():
            continue
        d = np.load(path)
        z, act = d['z'], d['action']          # (K,192), (K,10) z-scored blocks
        if len(z) < CTX + H:
            continue
        ctx = z[:CTX].copy()                  # real context z[0..2]
        real = z[CTX:CTX + H]                 # real next 15 frames
        a_blk = act[:CTX + H]                 # blocks leaving z[0..] (window [t:t+3])

        # --- autoregressive generation, real action driving ---
        seq = [torch.from_numpy(ctx[i]).float().to(DEVICE) for i in range(CTX)]
        for t in range(H):
            emb = torch.stack(seq[-CTX:])[None]
            act_w = torch.from_numpy(a_blk[t:t + CTX]).float().to(DEVICE)[None]
            with torch.no_grad():
                act_emb = model.action_encoder(act_w)
                z_next = model.predict(emb, act_emb)[0, -1]
            seq.append(z_next)
        gen = np.stack([s.cpu().numpy() for s in seq])        # (3+15, 192)
        real_full = np.concatenate([ctx, real], axis=0)       # (3+15, 192)

        # features on both sequences (dual: f3 = action-blind F_z surprise)
        f1r, f2r, f3r = feat_on_seq(real_full, act, model, fz=fz)
        f1g, f2g, f3g = feat_on_seq(gen, act, model, fz=fz)

        rows_r, ss = make_rows(f1r, f2r, f3r, mu, sd)
        rows_g, _ = make_rows(f1g, f2g, f3g, mu, sd)
        pr = np.array(head_probs(head, rows_r))
        pg = np.array(head_probs(head, rows_g))

        results[eid] = {
            'real': {'median_p': float(np.median(pr)), 'mean_p': float(pr.mean()),
                     'frac_p_gt_0.5': float((pr > 0.5).mean()),
                     'max_p': float(pr.max()),
                     'f3_median': float(np.nanmedian(f3r[3:]))},
            'gen': {'median_p': float(np.median(pg)), 'mean_p': float(pg.mean()),
                    'frac_p_gt_0.5': float((pg > 0.5).mean()),
                    'max_p': float(pg.max()),
                    'f3_median': float(np.nanmedian(f3g[3:]))},
        }
        done += 1
        if done >= 3:
            break

    out = {'mode': 'dual' if dual else 'wm_f3', 'episodes': results}
    (RUNS / out_name).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()