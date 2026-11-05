"""Benchmark a single CEM solve time on the PushT protocol.

Mirrors e6_eval.py / eval_wm.py setup but uses a plain episodic env reset
instead of a dataset, so it does NOT need the 46GB pusht dataset. Measures
the wall-clock time of one CEMSolver.solve() (the same "CEM solve time" the
mac-side used: h=5 -> 6.2s, h=15 -> 20.7s).

Usage:
    STABLEWM_HOME=D:/stable_worldmodel python bench_solve.py
"""

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

import time

import torch
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm
from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
from stable_worldmodel.planning.solver import CEMSolver

DEVICE = 'cuda'
HORIZONS = [5, 10, 15]
N_TIMED = 5


def img_transform(dtype=torch.float32, size=224):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=size),
    ])


def build(horizon):
    model = swm.wm.utils.load_pretrained('pusht/weights.pt')
    model = model.to(DEVICE).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    config = swm.PlanConfig(
        horizon=horizon,
        receding_horizon=5,
        action_block=5,
        history_len=3,
    )
    objective = GoalMSE()
    cost = ShootingCostEvaluator(model, objective)
    solver = CEMSolver(
        cost=cost,
        batch_size=1,
        num_samples=300,
        var_scale=1.0,
        n_steps=30,
        topk=30,
        device=DEVICE,
        seed=42,
    )
    transform = {
        'pixels': img_transform(),
        'goal': img_transform(),
    }
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process={}, transform=transform,
    )
    return model, solver, policy


def main():
    print(f'[bench] torch={torch.__version__} cuda={torch.cuda.is_available()} '
          f'device={DEVICE}')
    for horizon in HORIZONS:
        model, solver, policy = build(horizon)
        world = swm.World(
            env_name='swm/PushT-v1',
            num_envs=1,
            image_shape=(224, 224),
            max_episode_steps=150,
        )
        world.set_policy(policy)
        world.reset(seed=42)

        # warm-up solve (also the CUDA sanity: forward + one CEM solve).
        # First solve also pays CUDA kernel autotune/compile cost.
        t0 = time.perf_counter()
        policy.get_action(world.infos)
        torch.cuda.synchronize()
        warm = time.perf_counter() - t0
        print(f'[bench] horizon={horizon} warmup {warm:.3f}s')

        # timed solves: flush the action buffer each iteration so get_action
        # triggers a full re-plan (bare CEM solve), not a cached pop.
        times = []
        for _ in range(N_TIMED):
            world.reset(seed=42)
            for buf in policy._action_buffer:
                buf.clear()
            if policy._next_init is not None:
                policy._next_init.zero_()
            t0 = time.perf_counter()
            policy.get_action(world.infos)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        times = sorted(times)
        print(f'[bench] horizon={horizon} solves={[f"{t:.3f}" for t in times]} '
              f'median={times[len(times) // 2]:.3f}s')
        world.close()


if __name__ == '__main__':
    main()