"""Evaluation: rollout success, block-pose linear probes, distractor robustness,
cooling curves (U & S vs training step). Every number here is evidence for or
against the free-energy principle."""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import BCDataset
from .env_utils import (COORD, ColorJitter, make_env, obs_tensor,
                        random_probe_set, rollout_record)
from .model import Encoder, FlowHead, PolicyHead, flow_sample
from .train import ARMS, device_auto

PROBE_NPZ = "data/probe_set.npz"


def load_policy(arm, out_root="results", device=None):
    """Returns (enc, head, act_std); act_std is None for MSE heads.
    Checkpoints without a "kind" key (pre-flow) fall back to MSE."""
    device = device or device_auto()
    ckpt = torch.load(os.path.join(out_root, arm, "policy.pt"), map_location=device)
    enc = Encoder().to(device)
    enc.load_state_dict(ckpt["enc"])
    act_std = None
    if ckpt.get("kind", "mse") == "flow":
        head = FlowHead().to(device)
        act_std = ckpt["act_std"].to(device)
    else:
        head = PolicyHead().to(device)
    head.load_state_dict(ckpt["head"])
    enc.eval()
    head.eval()
    return enc, head, act_std


@torch.no_grad()
def embed_stack(enc, stack, device):
    """stack: (2,56,56,3) uint8 -> z (d,)"""
    z = enc(obs_tensor(stack, device))[0]
    return z


def make_policy_fn(enc, head, device, act_std=None):
    def fn(stack, ag_norm):
        with torch.no_grad():
            z = enc(obs_tensor(stack, device))[0]
            ag = torch.from_numpy(ag_norm).float().to(device).unsqueeze(0)
            if act_std is None:
                delta = head(z.unsqueeze(0), ag)[0].cpu().numpy()  # (action - agent) / 512
            else:  # flow head: integrate the velocity field to one action sample
                delta = flow_sample(head, z.unsqueeze(0), ag, act_std)[0].cpu().numpy()
        # ag_norm = agent/512 - 0.5  =>  agent = (ag_norm + 0.5) * 512
        agent_pos = (ag_norm + 0.5) * COORD
        return agent_pos + delta * COORD
    return fn


@torch.no_grad()
def encode_batch(enc, imgs, device, bs=512, jitter_fn=None):
    """imgs: (N,2,56,56,3) uint8 -> (N,d) numpy."""
    zs = []
    for i in range(0, len(imgs), bs):
        x = np.asarray(imgs[i:i + bs], dtype=np.float32) / 255.0
        if jitter_fn is not None:
            # keep the (B, 2, H, W, 3) stack layout; both frames share one
            # appearance shift (time-consistent jitter, as in rollouts)
            x = np.stack([np.stack([jitter_fn(f[0]), jitter_fn(f[1])])
                          for f in imgs[i:i + bs]]).astype(np.float32) / 255.0
        x = torch.from_numpy(x).to(device)  # (B,2,56,56,3)
        B = x.shape[0]
        x = x.permute(0, 1, 4, 2, 3).reshape(B, -1, 56, 56)  # (B,6,56,56)
        with torch.no_grad():
            zs.append(enc(x).cpu().numpy())
    return np.concatenate(zs)


def ridge_r2(Z, Y, alpha=1.0):
    """Closed-form ridge regression; returns per-target R^2 (train/test split)."""
    n = len(Z)
    idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    Ztr = np.c_[Z[tr], np.ones(len(tr))]
    Zte = np.c_[Z[te], np.ones(len(te))]
    W = np.linalg.solve(Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1]), Ztr.T @ Y[tr])
    pred = Zte @ W
    ss_res = ((Y[te] - pred) ** 2).mean(axis=0)
    ss_tot = ((Y[te] - Y[te].mean(axis=0)) ** 2).mean(axis=0)
    return 1 - ss_res / (ss_tot + 1e-12)


def probe_eval(arm, device, jitter_fn=None, out_root="results"):
    enc, _, _ = load_policy(arm, out_root, device)
    if not os.path.exists(PROBE_NPZ):
        random_probe_set(out_path=PROBE_NPZ)
    d = np.load(PROBE_NPZ)
    Z = encode_batch(enc, d["imgs"][:4000], device, jitter_fn=jitter_fn)
    r2_block = ridge_r2(Z, d["block"][:4000])  # x, y, cos, sin
    r2_agent = ridge_r2(Z, d["agent"][:4000])
    return {"block_x": float(r2_block[0]), "block_y": float(r2_block[1]),
            "block_cos": float(r2_block[2]), "block_sin": float(r2_block[3]),
            "agent_x": float(r2_agent[0]), "agent_y": float(r2_agent[1]),
            "block_mean": float(r2_block[:4].mean())}


def rollout_eval(arm, n_episodes, max_steps, device, jitter_seed=None, out_root="results"):
    enc, head, act_std = load_policy(arm, out_root, device)
    fn = make_policy_fn(enc, head, device, act_std)
    env = make_env()
    jit = ColorJitter(seed=jitter_seed) if jitter_seed is not None else None
    res = rollout_record(env, fn, n_episodes, max_steps, jitter=jit)
    env.close()
    succ = float(np.mean([r["success"] for r in res]))
    cov = float(np.mean([r["max_coverage"] for r in res]))
    return {"success": succ, "coverage": cov, "n": len(res)}


def bc_data_probe(arm, device, out_root="results"):
    """Extra diagnostic: how well does z predict the *action* it was trained on
    via a linear map? Measures linear actionability of the embedding."""
    enc, _, _ = load_policy(arm, out_root, device)
    ds = BCDataset(seed=7, n_samples=4000)
    Z, A = [], []
    dl = DataLoader(ds, batch_size=512)
    for x, _, a in dl:
        with torch.no_grad():
            Z.append(enc(x.to(device)).cpu().numpy())
        A.append(a.numpy())
    Z = np.concatenate(Z)
    A = np.concatenate(A)
    r2 = ridge_r2(Z, A)
    return {"action_r2": float(r2.mean())}


def cooling_curves(out_root="results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm in ARMS:
        p = os.path.join(out_root, arm, "metrics.jsonl")
        if not os.path.exists(p):
            continue
        steps, us, ss, ts = [], [], [], []
        for line in open(p):
            d = json.loads(line)
            if d.get("stage") != "wm":
                continue
            steps.append(d["step"])
            us.append(d["u"])
            ss.append(d["s"])
            ts.append(d["T"])
        if not steps:
            continue
        label = {"bc": "pixel BC", "bc_flow": "BC + flow matching",
                 "jepa": "JEPA+VICReg (patches)",
                 "e2jepa": "E2-JEPA (free energy)", "energy_only": "energy only (T=0)"}[arm]
        axes[0].plot(steps, us, label=label)
        axes[1].plot(steps, ss, label=label)
        if arm == "e2jepa" and any(t > 0 for t in ts):
            ax2 = axes[1].twinx()
            ax2.plot(steps, ts, "k:", alpha=0.4, label="T (annealed)")
            ax2.set_ylabel("temperature T")
            ax2.legend(loc="lower right", fontsize=8)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("U (prediction energy, per-dim)")
    axes[0].set_title("Energy: prediction gets good")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("S (per-dim entropy, nats)")
    axes[1].set_title("Entropy: patches vs principle")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "cooling_curves.png"), dpi=140)
    plt.close(fig)


def bars(summary, out_root="results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in ARMS if a in summary and "rollout" in summary[a]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    names = {"bc": "pixel BC", "bc_flow": "BC + flow", "jepa": "JEPA+VICReg",
             "e2jepa": "E2-JEPA", "energy_only": "energy only"}
    xs = np.arange(len(arms))

    axes[0].bar(xs, [summary[a]["rollout"]["success"] for a in arms], color="#0FDC78")
    axes[0].set_xticks(xs, [names[a] for a in arms], fontsize=8)
    axes[0].set_ylabel("success rate")
    axes[0].set_title("PushT success")

    axes[1].bar(xs, [summary[a]["probe"]["block_mean"] for a in arms], color="#02743B")
    axes[1].set_xticks(xs, [names[a] for a in arms], fontsize=8)
    axes[1].set_ylabel("R^2")
    axes[1].set_title("block-pose linear probe (x,y,cos,sin)")

    width = 0.35
    clean = [summary[a]["rollout"]["success"] for a in arms]
    dirty = [summary[a]["rollout_distract"]["success"] for a in arms]
    axes[2].bar(xs - width / 2, clean, width, label="clean", color="#02743B")
    axes[2].bar(xs + width / 2, dirty, width, label="distractor", color="#80C1FF")
    axes[2].set_xticks(xs, [names[a] for a in arms], fontsize=8)
    axes[2].set_ylabel("success rate")
    axes[2].set_title("spurious-appearance robustness")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_root, "bars.png"), dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--out_root", default="results")
    p.add_argument("--skip_curves", action="store_true")
    args = p.parse_args()

    device = device_auto()
    summary_path = os.path.join(args.out_root, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}

    for arm in args.arms:
        print(f"[eval:{arm}] probing ...", flush=True)
        entry = summary.get(arm, {})
        entry["probe"] = probe_eval(arm, device, out_root=args.out_root)
        entry["action_probe"] = bc_data_probe(arm, device, out_root=args.out_root)
        print(f"[eval:{arm}] rollouts (clean) ...", flush=True)
        entry["rollout"] = rollout_eval(arm, args.episodes, args.max_steps, device,
                                        out_root=args.out_root)
        print(f"[eval:{arm}] rollouts (distractor) ...", flush=True)
        entry["rollout_distract"] = rollout_eval(
            arm, args.episodes, args.max_steps, device, jitter_seed=123,
            out_root=args.out_root)
        entry["probe_distract"] = probe_eval(arm, device,
                                             jitter_fn=ColorJitter(seed=123),
                                             out_root=args.out_root)
        summary[arm] = entry
        json.dump(summary, open(summary_path, "w"), indent=2)

    if not args.skip_curves:
        cooling_curves(args.out_root)
        bars(summary, args.out_root)
    print(json.dumps({a: {k: v for k, v in summary[a].items() if k != "probe_distract"}
                      for a in args.arms if a in summary}, indent=2))


if __name__ == "__main__":
    main()
