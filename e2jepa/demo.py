"""Watch a trained policy push: one mp4 per episode + printed stats.

Usage:
    python -m e2jepa.demo --arm e2jepa --episodes 5
    python -m e2jepa.demo --arm bc --episodes 5

Seeds default to the same range as eval (seed0=1000), so episodes here are
the same initial states behind the summary.json numbers. Videos land in
results/videos/{arm}_ep{i}_{success|fail}_cov{max_coverage}.mp4 and show
the goal zone (LightGreen T), the last commanded action (red cross), and
the agent/block state at 384x384.
"""
import argparse
import json
import os

import imageio.v3 as iio
import numpy as np

from .env_utils import COORD, agent_pos_norm, make_env, render_obs
from .eval import load_policy, make_policy_fn
from .train import ARMS, device_auto


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="e2jepa", choices=ARMS)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    device = device_auto()
    enc, head, act_std = load_policy(args.arm, device=device)
    fn = make_policy_fn(enc, head, device, act_std)
    env = make_env(viz_size=384)
    out_dir = os.path.join("results", "videos")
    os.makedirs(out_dir, exist_ok=True)

    stats = []
    for ep in range(args.episodes):
        env.reset(seed=args.seed0 + ep)
        prev = render_obs(env)
        frames, success, max_cov = [], False, 0.0
        for t in range(args.max_steps):
            cur = render_obs(env)
            stack = np.stack([prev, cur])
            a = fn(stack, agent_pos_norm(env))
            _, _, _, _, info = env.step(np.clip(a, 0, COORD).astype(np.float32))
            prev = cur
            max_cov = max(max_cov, float(info.get("coverage", 0.0)))
            frames.append(env.render())
            if info.get("is_success", False):
                success = True
                break
        tag = "success" if success else "fail"
        name = f"{args.arm}_ep{ep}_{tag}_cov{max_cov:.2f}.mp4"
        iio.imwrite(os.path.join(out_dir, name), np.stack(frames),
                    plugin="FFMPEG", fps=args.fps)
        stats.append({"episode": ep, "success": success,
                      "max_coverage": round(max_cov, 3), "steps": t + 1})
        print(f"[{args.arm}] ep{ep}: steps={t + 1} "
              f"max_coverage={max_cov:.3f} success={success} -> {name}",
              flush=True)

    env.close()
    succ = float(np.mean([s["success"] for s in stats]))
    cov = float(np.mean([s["max_coverage"] for s in stats]))
    summary = {"arm": args.arm, "episodes": len(stats), "success": succ,
               "mean_max_coverage": cov, "runs": stats}
    out = os.path.join(out_dir, f"{args.arm}_demo.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"[{args.arm}] {len(stats)} eps: success={succ:.0%} "
          f"mean_max_coverage={cov:.3f}; videos in {out_dir}/")


if __name__ == "__main__":
    main()
