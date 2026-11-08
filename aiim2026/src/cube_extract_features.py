"""Cube twin of e2_extract_features.py (same f1/f2/f3 definition, same FS/W).

Differences from PushT:
  - dataset: official HF cube_single_expert.h5 (224x224 native, 10000 eps)
  - ckpt: cube/weights.pt (action dim 5 -> block 25)
  - 'state' column does not exist -> store qpos for downstream labelling
  - action has NaN at each episode's reset step -> nan_to_num(0) before z-score
Output: aiim2026/features_cube/{ep_id}.npz + meta.json
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

from e2_extract_features import causal_stats, episode_features, img_transform
import e2_extract_features as _e2

H5_PATH = Path(r'D:\stable_worldmodel\datasets\cube_single_expert.h5')
CKPT = 'cube/weights.pt'
OUT = Path(__file__).resolve().parents[1] / 'features_cube'
N_EPS, SEED = 1500, 0
FS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_e2.DEVICE = DEVICE   # episode_features uses the module-level device


def main(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = '_smoke' if smoke else ''
    n_eps = 3 if smoke else N_EPS

    model = swm.wm.utils.load_pretrained(CKPT).to(DEVICE).eval()
    model.requires_grad_(False)
    transform = img_transform()

    with h5py.File(H5_PATH, 'r', swmr=True,
                   rdcc_nbytes=256 * 1024 * 1024) as f:
        ep_len_all, ep_off_all = f['ep_len'][:], f['ep_offset'][:]
        act_col = f['action'][:]
        a_mean = np.nanmean(act_col, axis=0)
        a_std = np.nanstd(act_col, axis=0) + 1e-2
        a_dim = act_col.shape[1]
        del act_col

        rng = np.random.default_rng(SEED)
        eps = np.sort(rng.choice(len(ep_len_all), n_eps, replace=False))

        t0 = time.time()
        for n, ep in enumerate(eps):
            off, T = int(ep_off_all[ep]), int(ep_len_all[ep])
            K = T // FS
            pixels = f['pixels'][off:off + K * FS:FS]
            raw_act = f['action'][off:off + K * FS]
            raw_act = np.nan_to_num(raw_act, nan=0.0)
            qpos = f['qpos'][off:off + K * FS:FS]

            with torch.no_grad():
                frames = torch.from_numpy(pixels).permute(0, 3, 1, 2)
                frames = transform(frames)
                z = model.encode(
                    {'pixels': frames.unsqueeze(0).to(DEVICE)}
                )['emb'][0].float().cpu().numpy()

            act_z = (raw_act - a_mean) / a_std
            act_blocks = act_z.reshape(K, FS * a_dim)
            f1, f2, innov, stats = episode_features(z, act_blocks, model,
                                                    transform)
            np.savez_compressed(
                OUT / f'ep{ep}{suffix}.npz',
                z=z, f1=f1, f2=f2, f3=innov,
                action=act_blocks, qpos=qpos,
                ep_id=ep, raw_len=T, model_len=K, **stats,
            )
            if (n + 1) % 50 == 0:
                print(f'{n + 1}/{n_eps} episodes, {time.time() - t0:.1f}s',
                      flush=True)

    meta = {
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'experiment': 'cube feature extraction',
        'ckpt': CKPT, 'dataset': str(H5_PATH),
        'frameskip': FS, 'window': 8, 'n_episodes': n_eps, 'seed': SEED,
        'episode_ids': [int(e) for e in eps],
        'action_mean': a_mean.tolist(),
        'action_std_plus_eps': a_std.tolist(),
        'action_dim': int(a_dim),
        'feature_defs': {
            'f1': '||z_k - z_{k-1}||',
            'f2': 'cos(dz_k, dz_{k-1})',
            'f3': '||z_{k+1} - predict(z_{k-2..k}, a_{k-2..k})||',
            'f*_mean/std': 'causal window stats, W=8',
        },
        'smoke': smoke,
    }
    (OUT / f'meta{suffix}.json').write_text(json.dumps(meta, indent=2))
    print(f'done: {n_eps} eps in {time.time() - t0:.1f}s -> {OUT}')


if __name__ == '__main__':
    import sys
    main(smoke='--smoke' in sys.argv)
