"""BoundaryPolicy: WorldModelPolicy + online boundary detection head (E6).

Deployment of the E4 student exactly as it was trained:
  - per model step (= action_block raw steps): z_t from the frozen encoder
  - features on the latent increment stream: f1 ||dz||, f2 cos(dz_t, dz_{t-1}),
    f3 innovation ||z_{k+1} - p(z_{k-2..k}, a_{k-2..k})|| with actions
    z-scored by the E2 corpus stats
  - causal W=8 window stats -> 6 channels -> TinyGRU hidden state -> p
  - boundary arm: p > tau + 5-step refractory -> clear the action buffer ->
    this very get_action call replans from the freshest observation
  - equal arm: clear every `interval` model steps (frequency-matched control)
Feature indexing follows E2 exactly: f1 defined from step 1, f2 from 2,
f3 from 3; the GRU consumes one complete 6-channel row per model step
starting at step 3. All replans happen at block boundaries.
"""

from collections import deque

import numpy as np
import torch
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
from stable_worldmodel.policy import WorldModelPolicy

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_train_head import CausalCNN, TinyGRU

FS = 5           # action_block / frameskip (must match plan_config)
WINDOWS = (2, 4, 8)   # R2 multi-scale causal windows
TAU = 0.5
REFRACT = 5


class BoundaryPolicy(WorldModelPolicy):
    def __init__(self, *, solver, config, process, transform, model,
                 head_ckpt=None, action_mean=(0., 0.), action_std=(1., 1.),
                 arm='boundary', interval=4, head_class='gru',
                 channels=None, feat_mu=None, feat_sd=None,
                 tau=TAU, refractory=REFRACT):
        super().__init__(solver=solver, config=config, process=process,
                         transform=transform)
        self.model = model                    # frozen LeWM (eval, on device)
        self.arm = arm
        self.interval = max(1, int(interval))
        self.tau = tau
        self.refract = refractory
        self.head_class = head_class
        self.n_ch = len(channels) if channels else 6
        self.windows = WINDOWS if self.n_ch == 18 else (8,)
        if feat_mu is not None:
            self.feat_mu = np.asarray(feat_mu, np.float32)
            self.feat_sd = np.asarray(feat_sd, np.float32)
        else:
            self.feat_mu = None
        self.dev = next(model.parameters()).device
        self.head = None
        if head_ckpt is not None:
            head = (CausalCNN(in_ch=self.n_ch) if head_class == 'cnn'
                    else TinyGRU(in_ch=self.n_ch))
            head.load_state_dict(
                torch.load(head_ckpt, map_location='cpu'))
            head.eval()
            head.requires_grad_(False)
            head.to(self.dev)
            self.head = head
        self.a_mean = torch.tensor(action_mean, device=self.dev,
                                   dtype=torch.float32)
        self.a_std = torch.tensor(action_std, device=self.dev,
                                  dtype=torch.float32)
        self._frame_tf = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ])
        self.n_envs = 0
        self.st = []            # per-env detector state
        self.steps = []         # per-env executed raw steps
        self.episode_logs = []  # per-env list of completed episode logs

    # ------------------------------------------------------------------
    def _new_state(self):
        return {
            'z': deque(maxlen=4),          # z_{m-3..m} at boundary m
            'dz': deque(maxlen=2),         # last two increments (f2)
            'acts': deque(maxlen=FS * 3),  # last 3 raw action blocks
            'f1': [], 'f2': [], 'f3': [],  # raw features, one per boundary
            'rows': [],                     # emitted head rows (z-scored)
            'hidden': torch.zeros(1, 1, 16, device=self.dev),
            'm': -1,                       # model-step index of current obs
            'last_fire': -10**9,           # row index of last fire
            'since_replan': 0,
            'log': {'fires': [], 'replans': []},
        }

    def set_env(self, env):
        super().set_env(env)
        n = getattr(env, 'num_envs', 1)
        self.n_envs = n
        self.st = [self._new_state() for _ in range(n)]
        self.steps = [0] * n
        self.episode_logs = [[] for _ in range(n)]

    def _close_episode(self, i):
        st = self.st[i]
        if st['log']['replans'] or st['log']['fires']:
            self.episode_logs[i].append(st['log'])
        st['log'] = {'fires': [], 'replans': []}

    def finalize(self):
        """Call after evaluation: flush per-episode logs."""
        for i in range(self.n_envs):
            self._close_episode(i)
        return self.episode_logs

    # -- feature pipeline (mirrors e2_extract_features exactly) --------
    def _frame_tensor(self, pix_i):
        p = np.asarray(pix_i)
        if p.ndim == 4:                         # (T,H,W,C): take latest
            p = p[-1]
        x = torch.from_numpy(p).permute(2, 0, 1)   # HWC -> CHW
        return self._frame_tf(x)                    # (3,224,224)

    def _features(self, st, z_new):
        """Feed the new z at boundary m; emit the GRU row for step s=m.

        Mirrors E2 exactly: f1[m]=||z_m - z_{m-1}|| (m>=1), f2[m]=
        cos(dz_m, dz_{m-1}) (m>=2), f3[m]=||z_m - predict(z_{m-3..m-1},
        a_{m-3..m-1})|| (m>=3); row m uses causal W=8 window stats ending
        at index m; the GRU consumes rows m=3,4,5,... consecutively.
        """
        st['z'].append(z_new)
        st['m'] += 1
        m = st['m']
        # f1[m]
        if m >= 1:
            dz_new = st['z'][-1] - st['z'][-2]
            st['f1'].append(float(dz_new.norm().item()))
        else:
            dz_new = None
            st['f1'].append(np.nan)
        # f2[m]
        if m >= 2:
            prev = st['dz'][-1]
            cos = float(torch.dot(prev, dz_new).item()
                        / (prev.norm().item() * dz_new.norm().item() + 1e-12))
            st['f2'].append(cos)
        else:
            st['f2'].append(np.nan)
        if dz_new is not None:
            st['dz'].append(dz_new)
        # f3[m]: predict z_m from the 3 predecessor frames + 3 blocks
        if m >= 3 and len(st['acts']) >= FS * 3:
            emb_win = torch.stack(list(st['z'])[:3], dim=0)[None]  # (1,3,D)
            acts = torch.stack(
                [torch.as_tensor(a, device=self.dev, dtype=torch.float32)
                 for a in list(st['acts'])[-FS * 3:]])
            acts = (acts - self.a_mean) / self.a_std
            act_blocks = acts.reshape(1, 3, -1)                    # (1,3,10)
            with torch.no_grad():
                act_emb = self.model.action_encoder(act_blocks)
                pred = self.model.predict(emb_win, act_emb)        # (1,3,D)
                innov = float((st['z'][-1][None] - pred[:, -1]).norm().item())
            st['f3'].append(innov)                                 # index m
        else:
            st['f3'].append(np.nan)
        # emit row s = m: {f1,f2,f3} x {mean,std} x WINDOWS, matching the
        # offline R2 pipeline (r2_rescue.load_episode) channel for channel
        if m < 3:
            return None, None
        row = []
        for key in ('f1', 'f2', 'f3'):
            for w in self.windows:
                hist = [v for v in st[key][-w:] if np.isfinite(v)]
                if not hist:
                    return None, None
                arr = np.asarray(hist)
                row.append(float(arr.mean()))
                row.append(float(arr.std()) if len(arr) > 1 else 0.0)
        row = np.asarray(row, np.float32)
        if self.feat_mu is not None:
            row = (row - self.feat_mu) / self.feat_sd
        return m, row.astype(np.float32)

    def _gru_step(self, st, row):
        with torch.no_grad():
            x = torch.from_numpy(row).view(1, 1, -1).to(self.dev)
            out, h = self.head.gru(x, st['hidden'])
            p = float(torch.sigmoid(self.head.head(out)[0, -1]).item())
            st['hidden'] = h
            return p

    def _head_step(self, st, row):
        """One feature row through the deployed head. CNN: causal prefix
        pass over the emitted rows (position m output); GRU: hidden-state
        step. Both reproduce the offline full-sequence forward exactly."""
        if self.head_class == 'cnn':
            st['rows'].append(row)
            x = torch.from_numpy(np.stack(st['rows']))[None].to(self.dev)
            with torch.no_grad():
                logit = self.head(x)[0, -1]
            return float(torch.sigmoid(logit).item())
        return self._gru_step(st, row)

    # -- main ----------------------------------------------------------
    def get_action(self, info_dict):
        # 1. episode flush: reset detector state for restarted envs
        flush = info_dict.get('_needs_flush')
        if flush is not None:
            for i in np.nonzero(np.asarray(flush, dtype=bool))[0]:
                self._close_episode(i)
                self.st[i] = self._new_state()
                self.steps[i] = 0

        terminated = info_dict.get('terminated')
        if terminated is not None:
            alive = ~np.asarray(terminated, dtype=bool)
        else:
            alive = np.ones(self.n_envs, dtype=bool)

        # 2. block boundaries: detection BEFORE the base call, so a fire
        #    clears the buffer and THIS call replans from the fresh obs.
        #    'decoupled' arm: no detection, no forced clears — the base
        #    policy already re-anchors every receding_horizon blocks
        #    (horizon decoupled from exec length); only replans are logged.
        due = [i for i in range(self.n_envs)
               if alive[i] and self.steps[i] % FS == 0]
        if due:
            z_all = None
            if self.arm == 'boundary':
                frames = torch.stack(
                    [self._frame_tensor(info_dict['pixels'][i])
                     for i in due])
                with torch.no_grad():
                    z_all = self.model.encode(
                        {'pixels': frames.unsqueeze(1).to(self.dev)}
                    )['emb'][:, -1]                   # (B, D)
            for bi, i in enumerate(due):
                st = self.st[i]
                fired = False
                if self.arm == 'boundary':
                    s, row = self._features(st, z_all[bi])
                    if row is not None:
                        p = self._head_step(st, row)
                        if p > self.tau and s - st['last_fire'] >= self.refract:
                            st['last_fire'] = s
                            st['log']['fires'].append(
                                {'step': s, 'p': round(p, 4)})
                            fired = True
                # one replan decision per boundary
                if fired:
                    if len(self._action_buffer[i]) > 0:
                        self._action_buffer[i].clear()
                    st['log']['replans'].append(
                        {'block': self.steps[i] // FS, 'type': 'boundary'})
                    st['since_replan'] = 0
                elif (self.arm == 'equal'
                        and st['since_replan'] + 1 >= self.interval):
                    if len(self._action_buffer[i]) > 0:
                        self._action_buffer[i].clear()
                    st['log']['replans'].append(
                        {'block': self.steps[i] // FS, 'type': 'periodic'})
                    st['since_replan'] = 0
                elif len(self._action_buffer[i]) == 0:
                    st['log']['replans'].append(
                        {'block': self.steps[i] // FS, 'type': 'natural'})
                    st['since_replan'] = 0
                else:
                    st['since_replan'] += 1

        # 3. base class: flush handling, replanning, action pop, inverse tf
        action = super().get_action(info_dict)

        # 4. track executed raw actions / step counts
        for i in range(self.n_envs):
            if alive[i]:
                self.steps[i] += 1
                self.st[i]['acts'].append(
                    np.asarray(action[i], dtype=np.float32).reshape(-1))
        return action
