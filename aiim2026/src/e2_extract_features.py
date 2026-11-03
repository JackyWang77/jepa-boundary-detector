"""E2: extract three feature streams on the model-step time base.

Per EXPERIMENT_GUIDE.md (fixed design, do not modify):
  f1  increment energy     ||dz_k||                      + causal W=8 stats
  f2  direction deflection  cos(dz_k, dz_{k-1})          + causal W=8 stats
  f3  innovation            ||z_{k+1} - p(z_{k-2..k}, a_{k-2..k})|| + stats
Model-step time base (matches training): model frame k = raw frame 5*k,
action block k = raw actions [5k .. 5k+4] flattened (10-dim), full-column
z-scored with the training convention (std + 1e-2).
Output: aiim2026/features/{ep_id}.npz + meta.json (or *_smoke for --smoke).
"""

import json
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms

H5_PATH = Path.home() / '.stable_worldmodel/datasets/pusht_expert_train.h5'
CKPT = 'pusht/weights.pt'
OUT = Path(__file__).resolve().parents[1] / 'features'
N_EPS, SEED = 500, 0
FS = 5              # frameskip = action_block (training & eval config)
W = 8               # causal window for feature stats
CTX = 3             # predictor context (history_size)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def img_transform():
    return transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])


def causal_stats(x, w):
    """Causal sliding mean/std over [max(0, k-w+1) .. k], NaN-aware."""
    K = len(x)
    valid = ~np.isnan(x)
    xv = np.where(valid, x, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(xv)])
    cs2 = np.concatenate([[0.0], np.cumsum(xv * xv)])
    cn = np.concatenate([[0], np.cumsum(valid)]).astype(float)
    idx = np.arange(K)
    s = np.maximum(0, idx - w + 1)
    cnt = cn[idx + 1] - cn[s]
    s1 = cs[idx + 1] - cs[s]
    s2 = cs2[idx + 1] - cs2[s]
    mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
    var = np.where(cnt > 1,
                   s2 / np.maximum(cnt, 1) - mean ** 2, np.nan)
    std = np.where(var > 0, np.sqrt(var), 0.0)
    return mean, std


def episode_features(z, act_blocks, model, transform):
    """z: (K,192) numpy; act_blocks: (K,10) numpy (already z-scored)."""
    K = len(z)
    # ---- f3 innovation: 3-frame causal windows through the predictor ----
    n_win = K - CTX          # window i: ctx frames [i..i+2] -> z[i+3]
    innov = np.full(K, np.nan)
    if n_win > 0:
        idx = np.arange(n_win)
        win_idx = idx[:, None] + np.arange(CTX)       # (n_win, 3)
        tgt = torch.from_numpy(z[idx + CTX]).float().to(DEVICE)
        preds = []
        with torch.no_grad():
            for lo in range(0, n_win, 256):
                hi = min(lo + 256, n_win)
                emb = torch.from_numpy(z[win_idx[lo:hi]]).float().to(DEVICE)
                act = torch.from_numpy(
                    act_blocks[win_idx[lo:hi]]).float().to(DEVICE)
                act_emb = model.action_encoder(act)   # (B, 3, 192)
                p = model.predict(emb, act_emb)        # (B, 3, 192)
                preds.append(p[:, -1])
        preds = torch.cat(preds)
        innov[CTX:] = (tgt - preds).norm(dim=1).cpu().numpy()

    # ---- f1 / f2 on the increment stream ----
    dz = np.diff(z, axis=0)                            # (K-1, 192)
    f1 = np.full(K, np.nan)
    f1[1:] = np.linalg.norm(dz, axis=1)
    f2 = np.full(K, np.nan)
    if K > 2:
        n0 = np.linalg.norm(dz[:-1], axis=1)
        n1 = np.linalg.norm(dz[1:], axis=1)
        cos = np.sum(dz[1:] * dz[:-1], axis=1) / (n0 * n1 + 1e-12)
        f2[2:] = cos

    # ---- causal window stats ----
    stats = {}
    for name, raw in [('f1', f1), ('f2', f2), ('f3', innov)]:
        m, s = causal_stats(raw, W)
        stats[f'{name}_mean'], stats[f'{name}_std'] = m, s
    return f1, f2, innov, stats


def main(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = '_smoke' if smoke else ''
    n_eps = 2 if smoke else N_EPS

    model = swm.wm.utils.load_pretrained(CKPT)
    model = model.to(DEVICE).eval()
    model.requires_grad_(False)
    transform = img_transform()

    with h5py.File(H5_PATH, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024) as f:
        ep_len_all, ep_off_all = f['ep_len'][:], f['ep_offset'][:]
        # action stats over the FULL column (training convention: std + 1e-2)
        act_col = f['action'][:]
        a_mean = np.nanmean(act_col, axis=0)
        a_std = np.nanstd(act_col, axis=0) + 1e-2
        del act_col

        rng = np.random.default_rng(SEED)
        eps = np.sort(rng.choice(len(ep_len_all), n_eps, replace=False))

        t0 = time.time()
        for n, ep in enumerate(eps):
            off, T = int(ep_off_all[ep]), int(ep_len_all[ep])
            K = T // FS
            pixels = f['pixels'][off:off + K * FS:FS]      # (K,224,224,3)
            raw_act = f['action'][off:off + K * FS]        # (K*FS, 2)
            state = f['state'][off:off + K * FS:FS]        # (K, 7)

            with torch.no_grad():
                frames = torch.from_numpy(pixels).permute(0, 3, 1, 2)
                frames = transform(frames)
                z = model.encode(
                    {'pixels': frames.unsqueeze(0).to(DEVICE)}
                )['emb'][0].float().cpu().numpy()

            act_z = (raw_act - a_mean) / a_std
            act_blocks = act_z.reshape(K, FS * raw_act.shape[1])
            f1, f2, innov, stats = episode_features(z, act_blocks, model,
                                                    transform)
            np.savez_compressed(
                OUT / f'ep{ep}{suffix}.npz',
                z=z, f1=f1, f2=f2, f3=innov,
                action=act_blocks, state=state,
                ep_id=ep, raw_len=T, model_len=K, **stats,
            )
            if (n + 1) % 50 == 0:
                print(f'{n + 1}/{n_eps} episodes, {time.time() - t0:.1f}s')

    meta = {
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'experiment': 'E2 feature extraction',
        'ckpt': CKPT,
        'frameskip': FS,
        'window': W,
        'n_episodes': n_eps,
        'seed': SEED,
        'episode_ids': [int(e) for e in eps],
        'action_mean': a_mean.tolist(),
        'action_std_plus_eps': a_std.tolist(),
        'feature_defs': {
            'f1': '||z_k - z_{k-1}|| (model-step increments)',
            'f2': 'cos(dz_k, dz_{k-1})',
            'f3': '||z_{k+1} - predict(z_{k-2..k}, a_{k-2..k})||',
            'f*_mean/std': f'causal window stats, W={W}',
        },
        'smoke': smoke,
    }
    with open(OUT / f'meta{suffix}.json', 'w') as fp:
        json.dump(meta, fp, indent=2)
    print(f'done: {n_eps} episodes in {time.time() - t0:.1f}s -> {OUT}/*{suffix}*')


if __name__ == '__main__':
    import sys
    main(smoke='--smoke' in sys.argv)
