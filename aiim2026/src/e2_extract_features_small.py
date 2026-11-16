"""Phase 3: extract f1/f2/f3 features from the REDUCED-CAPACITY model.

Mirrors e2_extract_features.py exactly, but:
  - CKPT = pusht_small/weights.pt
  - OUT  = features_small/
  - DEVICE = cuda
  - h5 path from STABLEWM_HOME
So the train/hold episode split (e4_meta.json) and event labels are identical
to the main-model feature set; only the latent stream z differs.
"""

import json
import os
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

os.environ.setdefault('STABLEWM_HOME', r'D:\stable_worldmodel')
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms

SWM = Path(os.environ.get('STABLEWM_HOME', r'D:\stable_worldmodel'))
H5_PATH = SWM / 'datasets' / 'pusht_expert_train.h5'
CKPT = 'pusht_small/weights.pt'
OUT = Path(__file__).resolve().parents[1] / 'features_small'
N_EPS, SEED = 500, 0
FS = 5
W = 8
CTX = 3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def img_transform():
    return transforms.Compose([
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])


def causal_stats(x, w):
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


def episode_features(z, act_blocks, model):
    K = len(z)
    n_win = K - CTX
    innov = np.full(K, np.nan)
    if n_win > 0:
        idx = np.arange(n_win)
        win_idx = idx[:, None] + np.arange(CTX)
        tgt = torch.from_numpy(z[idx + CTX]).float().to(DEVICE)
        preds = []
        with torch.no_grad():
            for lo in range(0, n_win, 256):
                hi = min(lo + 256, n_win)
                emb = torch.from_numpy(z[win_idx[lo:hi]]).float().to(DEVICE)
                act = torch.from_numpy(
                    act_blocks[win_idx[lo:hi]]).float().to(DEVICE)
                act_emb = model.action_encoder(act)
                p = model.predict(emb, act_emb)
                preds.append(p[:, -1])
        preds = torch.cat(preds)
        innov[CTX:] = (tgt - preds).norm(dim=1).cpu().numpy()

    dz = np.diff(z, axis=0)
    f1 = np.full(K, np.nan)
    f1[1:] = np.linalg.norm(dz, axis=1)
    f2 = np.full(K, np.nan)
    if K > 2:
        n0 = np.linalg.norm(dz[:-1], axis=1)
        n1 = np.linalg.norm(dz[1:], axis=1)
        cos = np.sum(dz[1:] * dz[:-1], axis=1) / (n0 * n1 + 1e-12)
        f2[2:] = cos

    stats = {}
    for name, raw in [('f1', f1), ('f2', f2), ('f3', innov)]:
        m, sd = causal_stats(raw, W)
        stats[f'{name}_mean'], stats[f'{name}_std'] = m, sd
    return f1, f2, innov, stats


def main(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    n_eps = 2 if smoke else N_EPS

    model = swm.wm.utils.load_pretrained(CKPT)
    model = model.to(DEVICE).eval()
    model.requires_grad_(False)
    transform = img_transform()

    with h5py.File(H5_PATH, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024) as f:
        ep_len_all, ep_off_all = f['ep_len'][:], f['ep_offset'][:]
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
            pixels = f['pixels'][off:off + K * FS:FS]
            raw_act = f['action'][off:off + K * FS]
            state = f['state'][off:off + K * FS:FS]

            with torch.no_grad():
                frames = torch.from_numpy(pixels).permute(0, 3, 1, 2)
                frames = transform(frames)
                z = model.encode(
                    {'pixels': frames.unsqueeze(0).to(DEVICE)}
                )['emb'][0].float().cpu().numpy()

            act_z = (raw_act - a_mean) / a_std
            act_blocks = act_z.reshape(K, FS * raw_act.shape[1])
            f1, f2, innov, stats = episode_features(z, act_blocks, model)
            np.savez_compressed(
                OUT / f'ep{ep}.npz',
                z=z, f1=f1, f2=f2, f3=innov,
                action=act_blocks, state=state,
                ep_id=ep, raw_len=T, model_len=K, **stats,
            )
            if (n + 1) % 50 == 0:
                print(f'{n + 1}/{n_eps} episodes, {time.time() - t0:.1f}s',
                      flush=True)

    meta = {
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'experiment': 'E2 feature extraction (small capacity model)',
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
    }
    with open(OUT / 'meta.json', 'w') as fp:
        json.dump(meta, fp, indent=2)
    print(f'done: {n_eps} episodes in {time.time() - t0:.1f}s -> {OUT}',
          flush=True)


if __name__ == '__main__':
    import sys
    main(smoke='--smoke' in sys.argv)
