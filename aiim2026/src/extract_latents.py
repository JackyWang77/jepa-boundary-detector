"""Extract LeWM latents + states for 400 PushT episodes (300 train / 100 hold).

Pixel transform is identical to eval_wm.py (Mind the Gap protocol):
ToImage -> ToDtype(float32, scale=True) -> ImageNet Normalize -> Resize(224).
Outputs: runs/latents_{train,hold}.npz with flattened z / state and ep_len.
"""

import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import stable_worldmodel as swm
import stable_pretraining as spt
import torch
from torchvision.transforms import v2 as transforms

H5_PATH = Path.home() / '.stable_worldmodel/datasets/pusht_expert_train.h5'
CKPT = 'pusht/weights.pt'
OUT = Path(__file__).resolve().parents[1] / 'runs'
N_TOTAL, N_TRAIN, SEED = 400, 300, 0
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'


def img_transform():
    """Equivalent to eval_wm.py's pipeline; ToImage is done manually so the
    transform accepts a batched (T,H,W,C) numpy array."""
    return transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ]
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    transform = img_transform()

    model = swm.wm.utils.load_pretrained(CKPT)
    model = model.to(DEVICE).eval()
    model.requires_grad_(False)

    with h5py.File(H5_PATH, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024) as f:
        ep_len_all = f['ep_len'][:]
        ep_offset_all = f['ep_offset'][:]

        rng = np.random.default_rng(SEED)
        eps = rng.choice(len(ep_len_all), N_TOTAL, replace=False)
        eps = np.sort(eps)

        z_parts, s_parts = {0: [], 1: []}, {0: [], 1: []}
        lens = {0: [], 1: []}
        t0 = time.time()
        for i, ep in enumerate(eps):
            split = 0 if i < N_TRAIN else 1
            off, n = int(ep_offset_all[ep]), int(ep_len_all[ep])
            pixels = f['pixels'][off:off + n]          # (T,224,224,3) uint8
            state = f['state'][off:off + n].astype(np.float32)

            with torch.no_grad():
                # ToImage equivalent: HWC uint8 -> CHW tensor, batched on dim 0
                frames = torch.from_numpy(pixels).permute(0, 3, 1, 2)
                frames = transform(frames)              # (T,3,224,224)
                emb = model.encode({'pixels': frames.unsqueeze(0).to(DEVICE)})['emb']
            z_parts[split].append(emb[0].float().cpu().numpy())
            s_parts[split].append(state)
            lens[split].append(n)

            if (i + 1) % 50 == 0:
                print(f'{i + 1}/{N_TOTAL} episodes, {time.time() - t0:.1f}s')

    for split, name in [(0, 'train'), (1, 'hold')]:
        z = np.concatenate(z_parts[split], axis=0)
        s = np.concatenate(s_parts[split], axis=0)
        l = np.array(lens[split], dtype=np.int64)
        assert len(z) == len(s) == int(l.sum())
        np.savez_compressed(
            OUT / f'latents_{name}.npz', z=z, state=s, ep_len=l
        )
        print(f'[{name}] episodes={len(l)} frames={len(z)} '
              f'z={z.shape} state={s.shape}')

    print(f'done in {time.time() - t0:.1f}s -> {OUT}')


if __name__ == '__main__':
    main()
