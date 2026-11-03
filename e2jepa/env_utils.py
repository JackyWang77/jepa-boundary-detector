"""gym-pusht utilities: rendering, obs processing, probe labels, distractors."""
import numpy as np
import torch
from PIL import Image

import gymnasium as gym
import gym_pusht  # noqa: F401  registers the env

OBS_SIZE = 56
COORD = 512.0  # action/state coordinate space is [0,512]^2


def make_env(seed=None, viz_size=96):
    env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array",
                   visualization_width=viz_size, visualization_height=viz_size)
    env.reset(seed=seed)
    return env


def render_obs(env):
    """Observation render (96x96, same path as the teleop dataset), resized to 56."""
    img = env.unwrapped._render(visualize=False)  # noqa: SLF001 same as dataset
    pil = Image.fromarray(img).resize((OBS_SIZE, OBS_SIZE), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def obs_tensor(obs_stack, device):
    """(2,56,56,3) uint8 stack -> (1,6,56,56) float tensor in [0,1]."""
    x = torch.from_numpy(np.asarray(obs_stack, dtype=np.float32) / 255.0)
    x = x.permute(0, 3, 1, 2).reshape(1, 6, OBS_SIZE, OBS_SIZE)
    return x.to(device)


def agent_pos(env):
    return np.asarray(env.unwrapped.agent.position, dtype=np.float32)


def agent_pos_norm(env):
    return agent_pos(env) / COORD - 0.5


class ColorJitter:
    """Spurious-appearance shift: per-channel gain/bias, fixed for one episode.

    Models 'a new environment with same physics but different looks' —
    end-to-end pixel models absorb appearance, principled abstractions should not.
    """

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.gain = rng.uniform(0.6, 1.4, size=(3, 1, 1)).astype(np.float32)
        self.bias = rng.uniform(-30, 30, size=(3, 1, 1)).astype(np.float32)

    def __call__(self, img):
        x = img.astype(np.float32).transpose(2, 0, 1) * self.gain + self.bias
        return np.clip(x, 0, 255).transpose(1, 2, 0).astype(np.uint8)


def rollout_record(env, policy_fn, n_episodes, max_steps, jitter=None, seed0=1000):
    """Run policy in env. policy_fn(obs_stack, agent_norm) -> action np(2,).

    Returns list of dicts with success / max coverage.
    """
    out = []
    for ep in range(n_episodes):
        env.reset(seed=seed0 + ep)
        prev = render_obs(env)
        if jitter is not None:
            prev = jitter(prev)
        success, max_cov = False, 0.0
        for t in range(max_steps):
            cur = render_obs(env)
            if jitter is not None:
                cur = jitter(cur)
            stack = np.stack([prev, cur])
            a = policy_fn(stack, agent_pos_norm(env))
            _, _, _, _, info = env.step(np.clip(a, 0, COORD).astype(np.float32))
            prev = cur
            max_cov = max(max_cov, float(info.get("coverage", 0.0)))
            if info.get("is_success", False):
                success = True
                break
        out.append({"success": success, "max_coverage": max_cov, "steps": t + 1})
    return out


def random_probe_set(n_episodes=60, max_steps=120, seed0=50000, out_path=None):
    """Random rollouts -> labeled frames for linear probing of block pose.

    Alignment: stack (s_t, s_{t+1}) is labeled with s_{t+1}, the state shown
    in its SECOND frame - the same convention as the teleop encoding, where
    z_t encodes the (t-1, t) pair and represents state t. (An earlier
    version labeled the (t-1, t) stack with the post-step pose s_{t+1} -
    one step ahead of the image; fixed after reading gym_pusht's step().)
    """
    env = make_env()
    env.action_space.seed(seed0)
    imgs, block, agent = [], [], []
    for ep in range(n_episodes):
        env.reset(seed=seed0 + ep)
        prev = render_obs(env)
        for _ in range(max_steps):
            _, _, _, _, info = env.step(
                env.action_space.sample().astype(np.float32))
            cur = render_obs(env)
            imgs.append(np.stack([prev, cur]))
            bp = info["block_pose"]
            block.append([bp[0] / COORD, bp[1] / COORD,
                         np.cos(bp[2]), np.sin(bp[2])])
            agent.append(agent_pos_norm(env))
            prev = cur
    env.close()
    imgs = np.stack(imgs).astype(np.uint8)
    block = np.asarray(block, dtype=np.float32)
    agent = np.asarray(agent, dtype=np.float32)
    if out_path:
        np.savez_compressed(out_path, imgs=imgs, block=block, agent=agent)
    return imgs, block, agent
