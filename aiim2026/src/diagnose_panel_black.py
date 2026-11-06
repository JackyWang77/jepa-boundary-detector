"""Reproduce save_panel_videos with real data to find where panels go black."""
import os
os.environ.setdefault('STABLEWM_HOME', r'D:\stable_worldmodel')

import imageio.v2 as iio
import numpy as np
import stable_worldmodel as swm
from stable_worldmodel.plot import save_panel_videos

ds = swm.data.load_dataset(
    'ogbench/cube_single_expert.h5',
    keys_to_cache=['action', 'qpos', 'qvel'],
)

# mimic _extract_init_goal for 2 envs
data = ds.load_chunk(np.array([0, 1]), np.array([10, 10]),
                     np.array([36, 36]))
dataset_videos, goals = [], []
for ep in data:
    px = ep['pixels']
    if px.shape[1] == 3:            # CHW -> HWC as world.py does
        px = px.permute(0, 2, 3, 1)
    arr = px.numpy()
    print('dataset slice:', arr.shape, arr.dtype,
          'min/max/mean:', arr.min(), arr.max(), round(float(arr.mean()), 1))
    dataset_videos.append(arr)
    goals.append(arr[-1])

# fake agent frames from the real video (they were verified non-black)
frames0 = iio.mimread(r'..\runs\videos_cube\env_0.mp4')
PAD, W = 16, 224
agent = {i: [f[PAD:PAD + W, PAD:PAD + W] for f in frames0] for i in range(2)}

save_panel_videos(
    __import__('pathlib').Path(r'..\runs\videos_cube\_repro'),
    {'agent': agent, 'dataset': dataset_videos, 'goal': goals},
)
out = iio.mimread(r'..\runs\videos_cube\_repro\env_0.mp4')
mid = out[len(out) // 2]
H, Wf = mid.shape[:2]
for name, cx in (('left(agent)', Wf // 6), ('mid(dataset)', Wf // 2),
                 ('right(goal)', 5 * Wf // 6)):
    patch = mid[H // 4: 3 * H // 4, max(0, cx - 20): cx + 20]
    print(f'{name:14s} min={patch.min():3d} max={patch.max():3d} '
          f'mean={patch.mean():6.1f}')
