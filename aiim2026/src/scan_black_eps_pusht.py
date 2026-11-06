"""PushT dataset: same black-pixel scan (protects lane A/B/D validity)."""
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (pusht pixels are compressed)
import numpy as np

root = Path(r'D:\stable_worldmodel\datasets')
cands = list(root.rglob('*pusht*.h5')) + list(root.rglob('*PushT*.h5'))
print('candidates:', [str(c) for c in cands])

for h5p in cands:
    with h5py.File(h5p, 'r') as f:
        if 'pixels' not in f or 'ep_len' not in f:
            print(f'{h5p.name}: no pixels/ep_len, skip')
            continue
        ep_len = f['ep_len'][:]
        pixels = f['pixels']
        offs = np.concatenate([[0], np.cumsum(ep_len)])[:-1]
        n = len(ep_len)
        idx = np.arange(0, n, max(1, n // 200))       # sample ~200 eps
        black = 0
        for e in idx:
            mid = offs[e] + ep_len[e] // 2
            if int(pixels[mid].max()) == 0:
                black += 1
        print(json.dumps({
            'file': str(h5p), 'n_eps': int(n), 'sampled': len(idx),
            'black_frac': round(black / len(idx), 3),
        }))
