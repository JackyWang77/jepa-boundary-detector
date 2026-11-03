"""Train the arms:
  bc          pixel end-to-end BC (no world model)
  jepa        JEPA + VICReg patches (energy + second-order entropy surrogate)
  e2jepa      E2-JEPA: free energy F = U - T*S, temperature annealed
  energy_only JEPA with neither entropy nor patches (T=0) — collapse diagnostic
  tv          temperature vector: F = U - sum_k T_k S_k over K latent bands
              (the scale-resolved generalization of e2jepa; preregistered
              criteria scored in tau_spectrum.band_metrics)

Stages for jepa arms: (1) world-model pretraining on k-step latent prediction,
(2) freeze encoder, behavior-clone the policy head. The `bc` arm trains the
whole stack end-to-end on actions alone.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import BCDataset, WindowDataset, collate_windows, load_cache
from .loss import gaussian_entropy, temperature, vicreg_terms
from .model import (Encoder, FlowHead, PolicyHead, Predictor, emma_update,
                    make_target, rollout_latents)

ARMS = ["bc", "bc_flow", "jepa", "e2jepa", "energy_only", "tv"]

# tv arm: K temperature bands over contiguous latent blocks, band 0 = slow
# target (highest T_k). The geometric gradient 8:4:2:1 is normalized to sum
# to 1, so the total entropy budget matches the scalar e2jepa arm whenever
# the bands are decorrelated (the intended solution); early in training,
# cross-band correlations make the scalar's full-covariance S slightly
# smaller, so the tv budget is a slight overestimate that vanishes as the
# bands specialize. Only the DISTRIBUTION of entropy pressure across bands
# is the experimental variable.
N_BANDS = 4
BAND_MULT = [8.0, 4.0, 2.0, 1.0]
BAND_MULT = [m / sum(BAND_MULT) for m in BAND_MULT]


def device_auto():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def entropy_on_cpu(z, device):
    """slogdet on CPU for portability; gradient flows back across devices."""
    if z.device.type == "cpu":
        return gaussian_entropy(z)
    s = gaussian_entropy(z.cpu())
    return s.to(device)


def train_world_model(arm, out_dir, steps, batch, device, seed, log):
    torch.manual_seed(seed)
    np.random.seed(seed)

    enc = Encoder().to(device)
    pred = Predictor().to(device)
    tgt_enc = make_target(enc).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()), lr=3e-4)

    ds = WindowDataset(k_max=4, n_samples=steps * batch, seed=seed)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, collate_fn=collate_windows,
                    num_workers=0)

    step = 0
    enc.train()
    t0 = time.time()
    while step < steps:
        for x, a, mask, y, ks in dl:
            if step >= steps:
                break
            x, a, mask, y = x.to(device), a.to(device), mask.to(device), y.to(device)
            z = enc(x)
            with torch.no_grad():
                zt = tgt_enc(y)
            preds = rollout_latents(pred, z, a, int(ks.max()))
            # per-step energy: mean over batch, dims, horizon (masked)
            u = 0.0
            for i, zp in enumerate(preds):
                ki_mask = mask[:, i] > 0
                if ki_mask.any():
                    u = u + F.mse_loss(zp[ki_mask], zt[ki_mask])
            u = u / len(preds)

            sks = None
            t = 0.0
            loss = u
            if arm == "tv":
                w = z.shape[1] // N_BANDS
                t = temperature(step, steps)
                sks = []
                for k in range(N_BANDS):
                    sk = entropy_on_cpu(z[:, k * w:(k + 1) * w], device)
                    sks.append(float(sk.item()))
                    loss = loss - t * BAND_MULT[k] * sk
                s = sum(m * v for m, v in zip(BAND_MULT, sks))
            else:
                s = entropy_on_cpu(z, device).item() if arm == "e2jepa" \
                    else entropy_on_cpu(z.detach(), device).item()
                if arm == "e2jepa":
                    t = temperature(step, steps)
                    loss = u - t * entropy_on_cpu(z, device)
                elif arm == "jepa":
                    loss = u + vicreg_terms(z)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            emma_update(enc, tgt_enc, momentum=0.999)

            if step % 25 == 0:
                rec = {"stage": "wm", "step": step, "u": u.item(),
                       "s": s, "T": t, "loss": loss.item(),
                       "sec": round(time.time() - t0, 1)}
                if sks is not None:
                    rec["sk"] = [round(v, 4) for v in sks]
                log.write(json.dumps(rec) + "\n")
                log.flush()
            step += 1
    torch.save({"enc": enc.state_dict(), "pred": pred.state_dict()},
               os.path.join(out_dir, "world_model.pt"))
    return enc


def train_policy(arm, out_dir, steps, batch, device, seed, log, enc=None):
    """BC head on frozen (jepa arms) or end-to-end (bc/bc_flow arm) encoder.

    arm == "bc_flow": flow-matching head — regress the CondOT velocity field
    v(x_t, t | z, ag) to (a_s - x0) with a_s = delta / act_std, x0 ~ N(0,I).
    Otherwise: plain MSE regression on the delta target.
    Train uses SE(2) translation augmentation; every 10th episode is held out
    for a fixed validation batch tracked in metrics as stage "bcval".
    LR decays cosinely 3e-4 -> 3e-5 over the run.
    """
    torch.manual_seed(seed + 1)
    ds = BCDataset(seed=seed + 1, augment=True)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    vdl = DataLoader(BCDataset(seed=seed + 2, val=True), batch_size=batch,
                     shuffle=False, num_workers=0)
    val_batches = []
    for i, vb in enumerate(vdl):
        if i >= 8:
            break
        val_batches.append([t.to(device) for t in vb])
    flow = arm == "bc_flow"
    act_std = None
    if flow:
        # scale delta actions to ~unit std so the N(0,I) prior matches the data
        _, actions, agent, _, _, _ = load_cache()
        act_std = torch.from_numpy(((actions - agent) / 512.0).std(axis=0))
        act_std = act_std.float().clamp_min(1e-3).to(device)
        head = FlowHead().to(device)
        # fixed (x0, t) per val batch -> a deterministic val curve
        g = torch.Generator().manual_seed(123)
        for vb in val_batches:
            vb.append(torch.randn(vb[2].shape[0], 2, generator=g).to(device))
            vb.append(torch.rand(vb[2].shape[0], 1, generator=g).to(device))
    else:
        head = PolicyHead().to(device)
    if enc is not None:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)
        params = head.parameters()
    else:
        enc = Encoder().to(device)  # bc/bc_flow: fresh encoder, trained end-to-end
        params = list(head.parameters()) + list(enc.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4)

    def flow_mse(z, ag, a, x0=None, t=None):
        a_s = a / act_std
        if x0 is None:
            x0 = torch.randn_like(a_s)
            t = torch.rand(a_s.shape[0], 1, device=device)
        xt = (1 - t) * x0 + t * a_s
        return F.mse_loss(head(xt, t, z, ag), a_s - x0)

    def val_loss():
        tot, n = 0.0, 0
        with torch.no_grad():
            for vb in val_batches:
                x, ag, a = vb[0], vb[1], vb[2]
                z = enc(x)
                if flow:
                    lo = flow_mse(z, ag, a, x0=vb[3], t=vb[4])
                else:
                    lo = F.mse_loss(head(z, ag), a)
                tot += lo.item() * x.shape[0]
                n += x.shape[0]
        return tot / max(n, 1)

    step = 0
    t0 = time.time()
    while step < steps:
        for x, ag, a in dl:
            if step >= steps:
                break
            lr = 3e-4 * (0.1 + 0.45 * (1 + math.cos(math.pi * step / max(steps - 1, 1))))
            for g in opt.param_groups:
                g["lr"] = lr
            x, ag, a = x.to(device), ag.to(device), a.to(device)
            with torch.set_grad_enabled(enc.training):
                z = enc(x)
            loss = flow_mse(z, ag, a) if flow else F.mse_loss(head(z, ag), a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                log.write(json.dumps({"stage": "bc", "step": step,
                                      "loss": loss.item(),
                                      "sec": round(time.time() - t0, 1)}) + "\n")
                log.flush()
            if step % 1000 == 0 or step == steps - 1:
                log.write(json.dumps({"stage": "bcval", "step": step,
                                      "loss": round(val_loss(), 6),
                                      "sec": round(time.time() - t0, 1)}) + "\n")
                log.flush()
            step += 1
    ckpt = {"enc": enc.state_dict(), "head": head.state_dict(),
            "kind": "flow" if flow else "mse"}
    if flow:
        ckpt["act_std"] = act_std.cpu()
    torch.save(ckpt, os.path.join(out_dir, "policy.pt"))
    return enc, head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--wm_steps", type=int, default=3000)
    p.add_argument("--bc_steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_root", default="results")
    args = p.parse_args()

    device = device_auto()
    out_dir = os.path.join(args.out_root, args.arm)
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "metrics.jsonl"), "w")
    log.write(json.dumps({"arm": args.arm, "device": str(device)}) + "\n")

    enc = None
    if args.arm not in ("bc", "bc_flow"):
        enc = train_world_model(args.arm, out_dir, args.wm_steps, args.batch,
                                device, args.seed, log)
        enc = Encoder().to(device)
        enc.load_state_dict(torch.load(os.path.join(out_dir, "world_model.pt"),
                                       map_location=device)["enc"])
    train_policy(args.arm, out_dir, args.bc_steps, args.batch, device,
                 args.seed, log, enc=enc)
    log.close()
    print(f"[{args.arm}] done -> {out_dir}")


if __name__ == "__main__":
    main()
