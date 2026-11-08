"""ProbeTPolicy: the detector holds TWO posts, both dynamic, same dual head.

1) Planning post (IMAGINED stream): at every replan, one probe rollout from
   the 3 latest REAL latents (zero-action, PROBE_H blocks) is auto-regressed
   with the frozen LeWM. The first fire step s on this generated stream
   becomes the CEM horizon — "think up to where the model imagines an event".

2) Execution post (REAL stream): the detector also runs on the incoming real
   z stream (same 18-ch rows). A fire there means "an event is happening
   NOW" -> the remaining action buffer is cleared and an immediate replan is
   forced. So the actual walk length is the imagined s, truncated earlier by
   any real-stream fire: r <= h, and both are decided by the detector, one
   in imagination and one in reality.

The detector never enters the CEM loop, so candidates cannot game it.
warm_start is disabled (both lengths vary between calls).

Per-episode logs record every probe T' and every real-stream fire.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import stable_pretraining as spt
from stable_worldmodel.policy import WorldModelPolicy
from torchvision.transforms import v2 as transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_train_head import CausalCNN, TinyGRU

FS = 5
CTX = 3
PROBE_H = 10
WINDOWS = (2, 4, 8)
MIN_HORIZON = 5
ZC_MAX = 12            # real-stream buffer for the execution post


class ProbeTPolicy(WorldModelPolicy):
    """WorldModelPolicy + probe-T' horizon, zero in-CEM detector calls."""

    def __init__(self, *, solver, config, process, transform, model,
                 head_ckpt, channels, feat_mu, feat_sd, fz_path,
                 tau=0.5, refractory=5, action_mean=(0., 0.),
                 action_std=(1., 1.), head_class='cnn', t_plus_one=False):
        # warm-start off: T' varies, the leftover tail shape would not match.
        # Initial receding = PROBE_H so the action buffer (maxlen=
        # receding*action_block) is sized for the largest dynamic T'.
        config = replace(config, warm_start=False, receding_horizon=PROBE_H)
        super().__init__(solver=solver, config=config, process=process,
                         transform=transform)
        self.model = model
        self.dev = next(model.parameters()).device
        self.base_cfg = config
        self.tau = tau
        self.refractory = refractory
        # Ablation (default off): "T+1" — let the event block finish.
        # Planning post: horizon = T'+1 so the plan covers the detected
        # event block instead of stopping at its start.
        # Real-stream post: on fire, keep the current block's 5 actions
        # instead of dropping them, so the re-anchor happens AFTER the
        # event block rather than at its first step.
        self.t_plus_one = t_plus_one
        self.a_mean = torch.tensor(action_mean, device=self.dev,
                                   dtype=torch.float32)
        self.a_std = torch.tensor(action_std, device=self.dev,
                                  dtype=torch.float32)
        # flat dim of one CEM action block: n_action_dims * action_block
        # (PushT 2*5=10, cube 5*5=25) — was hard-coded to 10.
        self.act_flat_dim = (int(getattr(config, 'action_block', 5))
                             * len(action_mean))
        self.fz_W = torch.from_numpy(np.load(fz_path)).float().to(self.dev)
        self.feat_mu = torch.from_numpy(
            np.asarray(feat_mu, np.float32)).to(self.dev)
        self.feat_sd = torch.from_numpy(
            np.asarray(feat_sd, np.float32)).to(self.dev)

        head_cls = {'cnn': CausalCNN, 'gru': TinyGRU}[head_class]
        head = head_cls(in_ch=len(channels))
        head.load_state_dict(torch.load(head_ckpt, map_location='cpu'))
        head.eval()
        head.requires_grad_(False)
        self.head = head.to(self.dev)

        self._frame_tf = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ])

        self.n_envs = 0
        self.zc = []            # per-env deque of real boundary z's (maxlen 4)
        self.steps = []
        self.episode_logs = []

    # ------------------------------------------------------------------
    def set_env(self, env):
        super().set_env(env)
        self.n_envs = getattr(env, 'num_envs', 1)
        self.zc = [list() for _ in range(self.n_envs)]
        self.steps = [0] * self.n_envs
        self.last_fire_step = [-10 ** 9] * self.n_envs
        self.episode_logs = [[] for _ in range(self.n_envs)]
        self.cur_log = [{'probes': [], 'fires': []}
                        for _ in range(self.n_envs)]

    def _wrap(self, log):
        return {'probes': log['probes'],
                'replans': [{'block': p['block'], 'type': 'probe_t',
                             'T': p['T'], 'p': p['pmax']}
                            for p in log['probes']],
                'fires': [{'block': f['block'], 'type': 'real_fire',
                           'p': f['p']} for f in log.get('fires', [])]}

    def finalize(self):
        out = []
        for i in range(self.n_envs):
            if self.cur_log[i]['probes'] or self.cur_log[i]['fires']:
                self.episode_logs[i].append(self._wrap(self.cur_log[i]))
            self.cur_log[i] = {'probes': [], 'fires': []}
            out.append(self.episode_logs[i])
        return out

    # ------------------------------------------------------------------
    def _frame_z(self, pix_i):
        p = np.asarray(pix_i)
        if p.ndim == 4:
            p = p[-1]
        x = torch.from_numpy(p).permute(2, 0, 1)
        x = self._frame_tf(x)
        with torch.no_grad():
            emb = self.model.encode({'pixels': x[None, None].to(self.dev)})
        return emb['emb'][0, -1]                       # (192,)

    def _make_rows(self, f1, f2, f3):
        T = len(f1)
        rows, ss = [], []
        for s in range(3, T - 1):                      # f3[s] needs z[s+1]
            row = []
            for feat in (f1, f2, f3):
                for w in WINDOWS:
                    hist = [v for v in feat[max(0, s - w + 1):s + 1]
                            if np.isfinite(v)]
                    if not hist:
                        row.extend([0.0, 0.0])
                        continue
                    arr = np.asarray(hist)
                    row.append(float(arr.mean()))
                    row.append(float(arr.std()) if len(arr) > 1 else 0.0)
            row = (torch.tensor(row, device=self.dev) -
                   self.feat_mu) / self.feat_sd
            rows.append(row)
            ss.append(s)
        return rows, ss

    def _head_probs(self, rows):
        probs = []
        for i in range(len(rows)):
            x = torch.stack(rows[:i + 1])[None]
            with torch.no_grad():
                logit = self.head(x)[0, -1]
            probs.append(float(torch.sigmoid(logit).item()))
        return probs

    def _feat_on_probe(self, z_arr):
        """f1/f2/f3_dual on a full generated sequence (18, 192)."""
        T = len(z_arr)
        f1 = np.full(T, np.nan)
        f2 = np.full(T, np.nan)
        f3 = np.full(T, np.nan)
        dz = np.diff(z_arr, axis=0)
        f1[1:] = np.linalg.norm(dz, axis=1)
        if T > 2:
            n0 = np.linalg.norm(dz[:-1], axis=1) + 1e-12
            n1 = np.linalg.norm(dz[1:], axis=1) + 1e-12
            f2[2:] = np.sum(dz[1:] * dz[:-1], axis=1) / (n0 * n1)
        zt = torch.tensor(z_arr, device=self.dev)
        for t in range(3, T - 1):
            pred = (zt[t - CTX:t].ravel() @ self.fz_W).detach()
            f3[t] = float((zt[t + 1] - pred).norm().item())
        return f1, f2, f3

    def _probe_T(self, ctx_z):
        """One probe rollout from 3 real ctx latents -> first fire step T'."""
        seq = [z.clone() for z in ctx_z]
        with torch.no_grad():
            for _ in range(PROBE_H):
                emb = torch.stack(seq[-CTX:])[None]
                act = torch.zeros(1, CTX, self.act_flat_dim, device=self.dev)
                act_emb = self.model.action_encoder(act)
                seq.append(self.model.predict(emb, act_emb)[0, -1])
        z_arr = torch.stack(seq).cpu().numpy()          # (18, 192)
        f1, f2, f3 = self._feat_on_probe(z_arr)
        rows, ss = self._make_rows(f1, f2, f3)
        probs = self._head_probs(rows)
        last_fire = -10 ** 9
        t_fire = PROBE_H
        for s, p in zip(ss, probs):
            if p > self.tau and s - last_fire >= self.refractory:
                last_fire = s
                t_fire = s
        if self.t_plus_one and t_fire < PROBE_H:
            t_fire += 1
        return max(min(t_fire, PROBE_H), MIN_HORIZON), (
            float(max(probs)) if probs else 0.0)

    def _real_p(self, i):
        """Detector on the REAL stream: p at the current real step.
        Returns None while fewer than 5 real frames are buffered."""
        z_arr = torch.stack(self.zc[i]).cpu().numpy()
        f1, f2, f3 = self._feat_on_probe(z_arr)
        rows, _ = self._make_rows(f1, f2, f3)
        if not rows:
            return None
        return self._head_probs(rows)[-1]

    # ------------------------------------------------------------------
    def get_action(self, info_dict):
        flush = info_dict.get('_needs_flush')
        if flush is not None:
            for i in np.nonzero(np.asarray(flush, dtype=bool))[0]:
                if self.cur_log[i]['probes'] or self.cur_log[i]['fires']:
                    self.episode_logs[i].append(self._wrap(self.cur_log[i]))
                self.cur_log[i] = {'probes': [], 'fires': []}
                self.zc[i] = []
                self.steps[i] = 0
                self.last_fire_step[i] = -10 ** 9
        terminated = info_dict.get('terminated')
        alive = (~np.asarray(terminated, dtype=bool)
                 if terminated is not None else np.ones(self.n_envs, bool))

        for i in range(self.n_envs):
            if not alive[i] or self.steps[i] % FS != 0:
                continue
            z_i = self._frame_z(info_dict['pixels'][i])
            self.zc[i].append(z_i)
            if len(self.zc[i]) > ZC_MAX:
                self.zc[i].pop(0)
            # execution post: detector on the REAL stream. A fire means
            # "an event is happening NOW" -> drop the rest of the plan;
            # the empty buffer forces an immediate replan (walk truncated).
            if len(self._action_buffer[i]) > 0 and len(self.zc[i]) >= 5:
                p_now = self._real_p(i)
                # refractory is counted in detector ROWS (= blocks, 5 env
                # steps each), matching _probe_T and boundary_policy's
                # `s - last_fire >= refract`. Comparing raw env steps here
                # silently shrank it 5x (one row between fires).
                blk = self.steps[i] // FS
                last_blk = self.last_fire_step[i] // FS
                if (p_now is not None and p_now > self.tau
                        and blk - last_blk >= self.refractory):
                    self.last_fire_step[i] = self.steps[i]
                    if self.t_plus_one:
                        # finish the current block (FS actions), replan
                        # after the event instead of at its first step
                        buf = self._action_buffer[i]
                        keep = list(buf)[:FS]
                        buf.clear()
                        buf.extend(keep)
                    else:
                        self._action_buffer[i].clear()
                    self.cur_log[i]['fires'].append(
                        {'block': self.steps[i] // FS,
                         'p': round(p_now, 4)})
            # planning post: probe the IMAGINED stream only when this call
            # will actually replan (buffer empty, incl. after a real fire)
            if len(self._action_buffer[i]) == 0 and len(self.zc[i]) >= CTX:
                Tp, pmax = self._probe_T(self.zc[i][-CTX:])
                # imagined fire step s sets the horizon AND the nominal
                # walk length; a later real-stream fire truncates the walk.
                self.cfg = replace(self.base_cfg, horizon=int(Tp),
                                   receding_horizon=int(Tp))
                self.base_cfg = self.cfg
                self.solver._config = self.cfg
                self.cur_log[i]['probes'].append(
                    {'block': self.steps[i] // FS, 'T': int(Tp),
                     'pmax': round(pmax, 4)})

        action = super().get_action(info_dict)

        for i in range(self.n_envs):
            if alive[i]:
                self.steps[i] += 1
        return action