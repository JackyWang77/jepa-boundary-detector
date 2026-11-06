"""Scan all 500 cube episodes: how many have black pixels?

Samples the middle frame of every episode; an episode is 'black' when its
sampled frame is all-zero (or near-zero).
"""
import json

import h5py
import numpy as np

H5 = r'D:\stable_worldmodel\datasets\ogbench\cube_single_expert.h5'

with h5py.File(H5, 'r') as f:
    ep_len = f['ep_len'][:]
    pixels = f['pixels']
    offs = np.concatenate([[0], np.cumsum(ep_len)])[:-1]
    stats = []
    for e in range(len(ep_len)):
        mid = offs[e] + ep_len[e] // 2
        frame = pixels[mid]                     # (224, 224, 3)
        stats.append((e, float(frame.mean()), int(frame.max())))

means = np.array([s[1] for s in stats])
blacks = [e for e, m, mx in stats if mx == 0]
near = [e for e, m, mx in stats if mx > 0 and m < 1.0]
out = {
    'n_episodes': len(stats),
    'n_fully_black': len(blacks),
    'black_episode_ids_first20': blacks[:20],
    'n_near_black(max>0,mean<1)': len(near),
    'mean_of_means': float(means.mean()),
    'min_ep_mean': float(means.min()),
    'max_ep_mean': float(means.max()),
}
print(json.dumps(out, indent=2))
