"""PACE baseline arm: phase-aware chunk execution without a detector.

Re-implementation of PACE (Nie et al., arXiv:2606.00537) on the same CEM
planner that the flat and probe arms use, so the three arms differ only in
the truncation rule:

  * the planning horizon is fixed to PACE_H = 10 macro-blocks, the same upper
    bound the probe arm can select;
  * every CEM query returns the full chunk a_1..a_h, each a_i flattened over
    B primitive actions;
  * the block speed is s_i = mean_j ||a_ij||_2 in raw action units (the
    inverse of the per-dimension standardisation used by the planner);
  * the earliest block k in [min_blocks, h-1] with
        s_k <= theta * median(s_1..s_h)
    marks a low-speed phase transition: k blocks are executed and the plan is
    re-anchored; no such block exists -> the whole chunk is executed, which
    reproduces the flat h=10 arm;
  * the detector and the latent stream are never touched, matching the
    training-free and plug-and-play contract of PACE.

The chunk is intercepted with a thin proxy around the solver, so the CEM
search loop and the world-model rollout are untouched.
"""

from dataclasses import replace

import numpy as np

from stable_worldmodel.policy import WorldModelPolicy

PACE_H = 10


class _RecordingSolver:
    """Delegating solver proxy that keeps the last chunk it returned."""

    def __init__(self, solver):
        self._solver = solver
        self.last_actions = None

    def __getattr__(self, name):
        return getattr(self._solver, name)

    def __call__(self, info_dict, init_action=None):
        out = self._solver(info_dict, init_action=init_action)
        self.last_actions = out['actions']
        return out


class PACEPolicy(WorldModelPolicy):
    def __init__(self, *, solver, config, process, transform,
                 action_mean, action_std, theta=0.5, min_blocks=2):
        # Both lengths are fixed: PACE chooses the executed prefix per call,
        # the buffer is sized for the largest possible prefix.
        config = replace(config, warm_start=False, horizon=PACE_H,
                         receding_horizon=PACE_H)
        super().__init__(solver=solver, config=config, process=process,
                         transform=transform)
        self._rec = _RecordingSolver(solver)
        self.solver = self._rec
        self.theta = float(theta)
        self.min_blocks = max(1, int(min_blocks))
        self.action_block = int(getattr(config, 'action_block', 5))
        self.a_mean = np.asarray(action_mean, np.float32)
        self.a_std = np.asarray(action_std, np.float32)
        self.n_envs = 0
        self.steps = []
        self.logs = []
        self.cur = []

    # ------------------------------------------------------------------
    def set_env(self, env):
        super().set_env(env)
        self.n_envs = getattr(env, 'num_envs', 1)
        self.steps = [0] * self.n_envs
        self.logs = [[] for _ in range(self.n_envs)]
        self.cur = [[] for _ in range(self.n_envs)]

    def keep_blocks(self, chunk):
        """chunk: (h, B*D) numpy -> number of blocks to execute."""
        h = chunk.shape[0]
        d = self.a_mean.shape[0]
        b = max(1, chunk.shape[1] // d)
        prim = chunk.reshape(h, b, d) * self.a_std + self.a_mean
        speeds = np.linalg.norm(prim, axis=2).mean(axis=1)
        med = float(np.median(speeds))
        if med <= 0:
            return h
        lo = min(self.min_blocks, h)
        for k in range(lo, h):
            if speeds[k - 1] <= self.theta * med:
                return k
        return h

    def finalize(self):
        out = []
        for i in range(self.n_envs):
            if self.cur[i]:
                self.logs[i].append({'replans': list(self.cur[i]), 'fires': []})
            self.cur[i] = []
            out.append(self.logs[i])
        return out

    # ------------------------------------------------------------------
    def get_action(self, info_dict):
        n = self.n_envs
        flush = info_dict.get('_needs_flush')
        flushed = (np.asarray(flush, dtype=bool) if flush is not None
                   else np.zeros(n, bool))
        for i in range(n):
            if flushed[i]:
                self.cur[i] = []
                self.steps[i] = 0
        terminated = info_dict.get('terminated')
        dead = (np.asarray(terminated, dtype=bool) if terminated is not None
                else np.zeros(n, bool))
        replan = [i for i in range(n)
                  if (len(self._action_buffer[i]) == 0 or flushed[i])
                  and not dead[i]]

        self._rec.last_actions = None
        action = super().get_action(info_dict)
        chunk_all = self._rec.last_actions

        if chunk_all is not None and replan:
            for row, env_i in enumerate(replan):
                row_t = chunk_all[row]
                chunk = (row_t.detach().cpu().numpy()
                         if hasattr(row_t, 'detach') else np.asarray(row_t))
                keep = self.keep_blocks(chunk)
                buf = self._action_buffer[env_i]
                n_steps = keep * self.action_block
                kept = list(buf)[:max(0, n_steps - 1)]
                buf.clear()
                buf.extend(kept)
                self.cur[env_i].append(
                    {'block': self.steps[env_i] // self.action_block,
                     'keep': int(keep), 'h': int(chunk.shape[0]),
                     'theta': self.theta})

        for i in range(n):
            if not dead[i]:
                self.steps[i] += 1
        return action
