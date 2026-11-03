"""Teleop dataset: LeRobot pusht parquet+mp4 -> cached arrays -> window datasets."""
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .env_utils import OBS_SIZE

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pusht")
CACHE = os.path.join(DATA_DIR, "cache.npz")


def build_cache():
    """Decode all mp4 frames once -> obs56 uint8 + actions + episode table."""
    if os.path.exists(CACHE):
        return
    import imageio.v3 as iio

    df = pd.read_parquet(os.path.join(DATA_DIR, "data/chunk-000/file-000.parquet"))
    n = len(df)
    actions = np.asarray(df["action"].tolist(), dtype=np.float32)
    agent = np.asarray(df["observation.state"].tolist(), dtype=np.float32)
    cov = np.asarray(df["next.reward"].tolist(), dtype=np.float32)
    ep = np.asarray(df["episode_index"], dtype=np.int64)
    starts = np.array([np.searchsorted(ep, i) for i in range(ep[-1] + 1)])
    ends = np.array([np.searchsorted(ep, i, side="right") for i in range(ep[-1] + 1)])

    obs = np.zeros((n, OBS_SIZE, OBS_SIZE, 3), dtype=np.uint8)
    it = iio.imiter(os.path.join(DATA_DIR, "videos/observation.image/chunk-000/file-000.mp4"),
                    plugin="FFMPEG")
    for i, frame in enumerate(it):
        if i >= n:
            break
        pil = Image.fromarray(frame).resize((OBS_SIZE, OBS_SIZE), Image.BILINEAR)
        obs[i] = np.asarray(pil, dtype=np.uint8)
    assert i == n - 1, f"frame count mismatch: {i} vs {n}"

    np.savez_compressed(CACHE, obs=obs, actions=actions, agent=agent, cov=cov,
                        starts=starts, ends=ends)


def load_cache():
    if not os.path.exists(CACHE):
        build_cache()
    z = np.load(CACHE)
    return (z["obs"], z["actions"], z["agent"], z["starts"], z["ends"], z["cov"])


def _stack(obs, i_prev, i_cur):
    """Two-frame channel stack (crude velocity), clamped inside episode."""
    return np.concatenate([obs[i_prev], obs[i_cur]], axis=-1)  # (56,56,6)


class WindowDataset(Dataset):
    """World-model windows: (t, k) -> input stack, k actions, target stack.

    Latent k-step prediction: z_t --(a_t..a_{t+k-1})--> ẑ vs z̄_{t+k}.
    """

    def __init__(self, k_max=4, n_samples=100_000, seed=0):
        self.obs, self.actions, _, self.starts, self.ends, _ = load_cache()
        self.k_max = k_max
        rng = np.random.default_rng(seed)
        idxs = []
        n_ep = len(self.starts)
        for _ in range(n_samples):
            e = rng.integers(n_ep)
            s, e_end = self.starts[e], self.ends[e]
            L = e_end - s
            if L < 3:
                continue
            t = rng.integers(s, e_end - 1)
            k = int(rng.integers(1, min(k_max, e_end - 1 - t) + 1))
            idxs.append((t, k))
        self.idxs = np.asarray(idxs, dtype=np.int64)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t, k = self.idxs[i]
        s = np.searchsorted(self.ends, t, side="right") - 1
        s = self.starts[s]
        prev = max(t - 1, s)
        x = torch.from_numpy(_stack(self.obs, prev, t).astype(np.float32) / 255.0)  # (56,56,6)
        x = x.permute(2, 0, 1)
        a = torch.from_numpy(self.actions[t:t + k] / 512.0)  # (k,2)
        tgt_prev = max(t + k - 1, s)
        y = torch.from_numpy(_stack(self.obs, tgt_prev, t + k).astype(np.float32) / 255.0)
        y = y.permute(2, 0, 1)
        return x, a, y, k


def _shift(img, dx, dy):
    """Translate (H,W,C) image by (dx, dy) px, zero-fill borders. +dx moves content right."""
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    out[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        img[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
    return out


class BCDataset(Dataset):
    """(obs stack, agent_pos, action) for behavior cloning.

    augment: exploit the SE(2) translation symmetry of the table. Shift the
    image by (dx, dy) image-pixels and shift the agent/action world coords by
    the same motion rescaled to the 512-unit env frame. The delta target
    (action - agent) is invariant under this joint shift, so the policy is
    forced to learn push *geometry* rather than screen position.
    val: keep only every 10th episode (e % 10 == 9) as a held-out split.
    """

    def __init__(self, seed=1, n_samples=100_000, augment=False, max_shift=3,
                 val=False):
        self.obs, self.actions, self.agent, self.starts, self.ends, cov = load_cache()
        self.cov = cov
        rng = np.random.default_rng(seed)
        idxs = []
        for e in range(len(self.starts)):
            if (e % 10 == 9) != bool(val):
                continue
            idxs.extend(range(self.starts[e], self.ends[e] - 1))
        idxs = np.asarray(idxs, dtype=np.int64)
        if val:
            self.idxs = idxs  # holdout keeps the natural distribution
        else:
            # oversample endgame states: frames where the block already covers
            # much of the T are rare in teleop, but they are exactly where the
            # policy must be precise (the final rotational alignment).
            w = 1.0 + 4.0 * self.cov[idxs]
            self.idxs = rng.choice(idxs, size=n_samples, replace=True, p=w / w.sum())
        self.augment = augment
        self.max_shift = max_shift

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = self.idxs[i]
        s = np.searchsorted(self.ends, t, side="right") - 1
        s = self.starts[s]
        prev = max(t - 1, s)
        img = _stack(self.obs, prev, t)  # (56,56,6) uint8
        ag_px = self.agent[t].astype(np.float32)
        act_px = self.actions[t].astype(np.float32)
        if self.augment:
            dx = int(np.random.randint(-self.max_shift, self.max_shift + 1))
            dy = int(np.random.randint(-self.max_shift, self.max_shift + 1))
            if dx or dy:
                img = _shift(img, dx, dy)
                # one image pixel = 512/OBS_SIZE world units
                wdx = np.float32(dx * 512.0 / OBS_SIZE)
                wdy = np.float32(dy * 512.0 / OBS_SIZE)
                ag_px = ag_px + np.array([wdx, wdy], dtype=np.float32)
                act_px = act_px + np.array([wdx, wdy], dtype=np.float32)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        ag = torch.from_numpy(ag_px / 512.0 - 0.5)
        # Predict the *displacement* (action - agent_pos), not the absolute
        # target position. The absolute target is ~= current agent pos under PD
        # control, so predicting it directly lets the policy cheat by outputting
        # the current position -> agent never moves. Delta forces the policy to
        # use the image to decide *which way* to push.
        a = torch.from_numpy((act_px - ag_px) / 512.0)
        return x, ag, a


def collate_windows(batch):
    xs, acts, ys, ks = zip(*batch)
    k_max = max(ks)
    x = torch.stack(xs)
    y = torch.stack(ys)
    B = len(ks)
    a = torch.zeros(B, k_max, 2)
    mask = torch.zeros(B, k_max)
    for i, (ak, ki) in enumerate(zip(acts, ks)):
        a[i, :ki] = ak
        mask[i, :ki] = 1.0
    return x, a, mask, y, torch.tensor(ks)
