"""Convert the official OGBench cube val npz to the h5 layout the eval
envelope reads.

Data content is untouched -- this is a pure format adaptation:
  observations -> pixels (N, 64, 64, 3) uint8   (official 64x64 renders;
                                                 the eval pipeline resizes to
                                                 the model's 224 input)
  qpos/qvel/action            kept verbatim
  episode_idx/step_idx        derived from terminals
  ep_len/ep_offset            derived from terminals
  privileged_block_0_pos/quat sliced out of qpos (ADR=14), same as
                              patch_cube_h5.py did for the local dataset
"""
import json
from pathlib import Path

import h5py
import numpy as np

NPZ = Path(r'D:\stable_worldmodel\datasets\ogbench\official'
           r'\visual-cube-single-play-v0-val.npz')
OUT = Path(r'D:\stable_worldmodel\datasets\ogbench\cube_official_val.h5')
META = Path(r'd:\Research-2026-jepa-boundary-detector\aiim2026\runs'
            r'\cube_official_val.json')
ADR = 14  # object_joint_0 qpos address


def main():
    f = np.load(NPZ, allow_pickle=True)
    pix = f['observations']
    qpos = f['qpos']
    qvel = f['qvel']
    act = f['actions']
    term = f['terminals'].astype(bool)
    n = len(pix)
    assert pix.shape[1:] == (64, 64, 3), pix.shape

    ep_end = np.nonzero(term)[0]
    ep_start = np.concatenate([[0], ep_end[:-1] + 1])
    lens = (ep_end - ep_start + 1).astype(np.int64)
    num_ep = len(lens)
    assert lens.sum() == n, (lens.sum(), n)

    episode_idx = np.repeat(np.arange(num_ep, dtype=np.int64), lens)
    step_idx = np.concatenate([np.arange(k, dtype=np.int64) for k in lens])
    ep_offset = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)

    pos = np.ascontiguousarray(qpos[:, ADR:ADR + 3], dtype=np.float64)
    quat = np.ascontiguousarray(qpos[:, ADR + 3:ADR + 7], dtype=np.float64)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT, 'w') as h:
        h.create_dataset('pixels', data=pix, chunks=(644, 64, 64, 3))
        h.create_dataset('qpos', data=qpos)
        h.create_dataset('qvel', data=qvel)
        h.create_dataset('action', data=act)
        h.create_dataset('episode_idx', data=episode_idx)
        h.create_dataset('step_idx', data=step_idx)
        h.create_dataset('ep_len', data=lens.astype(np.int32))
        h.create_dataset('ep_offset', data=ep_offset)
        h.create_dataset('privileged_block_0_pos', data=pos)
        h.create_dataset('privileged_block_0_quat', data=quat)

    meta = {
        'source_npz': str(NPZ),
        'num_episodes': int(num_ep),
        'rows': int(n),
        'ep_len': int(lens[0]),
        'pixels': '64x64x3 uint8 (official render)',
        'out': str(OUT),
    }
    META.parent.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(META, 'w'), indent=2)
    print(f'wrote {OUT} ({OUT.stat().st_size/1e6:.0f}MB) rows={n} '
          f'eps={num_ep} len={lens[0]} -> {META}')


if __name__ == '__main__':
    main()
