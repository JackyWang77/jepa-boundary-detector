"""E5: MRBP v3.0 - does the multi-rate duty (A3) place content by tau?

Everything before this measured NECESSITY (self-organized codes never
assemble) and closed the budget pathway (v1: allocation, E1: total). The
only surviving placement force is A3: each block predicts ITS OWN future
at ITS OWN stride, so timescale-mismatched content raises that block's U
and is evicted by the gradient. Fast content is strongly evicted from
coarse blocks (its ACF dies inside the long window -> zero U credit,
pure entropy burden); fine blocks accept everything - so fast content
concentrates, and slow content must be retained by the slow head's
long-horizon U. A0 is implemented as grouped final LayerNorm (breaks the
Curie-Weiss all-to-all mixing that v1 died on); A2 as a fixed block
address space whose CONTENT routing stays learnable end-to-end through
the encoder; A4 as per-block logdet with a single shared annealed T
(total and allocation are proven insensitive - E1 / v1 - so T is not an
experimental variable here).

Preregistered criteria (fixed before training):
  C1  tau clustering: per-block median data-side tau ordered by stride
      (slow >= mid >= fast) AND slow/fast ratio >= 2.5 (all known arms
      are <= 1.03; the task spectrum is {~2, 13, 22} by the same
      estimator - E3 stroke-segmentation ground truth)
  C2  content placement: slow-block block-probe R2 minus fast-block
      >= 0.2, and fast-block agent-probe exceeds slow-block by >= 0.05
  C3  survival: every block's mean dim std >= 0.01
  Control arm: identical architecture, grouping and heads but ALL
  strides = 1 (single-rate). If the control also passes C1/C2, the
  clustering is an artifact of the grouped architecture, not of A3.

Arms:  mrpb_v3 (strides 5/2/1)  |  mrpb_single (strides 1/1/1)
Output: results/mrbp.json (+ ckpts for later audits), incremental.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_cache
from .eval import PROBE_NPZ, encode_batch, ridge_r2
from .loss import gaussian_entropy, temperature
from .model import emma_update, make_target
from .tau_spectrum import encode_episodes, tau_acf
from .train import device_auto

N_BLOCKS = 3
BLOCK_SIZES = [42, 42, 44]          # sum = 128, same latent width as e2jepa
STRIDES_V3 = [5, 2, 1]              # slow / mid / fast (task spectrum 22/13/2)
MULTS = [1, 2, 4]                   # horizons = stride * mult
STEPS = 3000
BATCH = 256
SEED = 0
OUT = os.path.join("results", "mrbp.json")
R2_GAP = 0.2                        # C2 threshold
TAU_RATIO = 2.5                     # C1 threshold

CUR_STRIDES = STRIDES_V3            # set per arm by train_arm


class GroupedEncoder(torch.nn.Module):
    """e2jepa's ConvNet trunk; the final LayerNorm is replaced by
    per-block LayerNorms - the minimal A0 local topology (v1's lesson:
    a global LN is all-to-all mean-field and cannot host multi-block
    phases)."""

    def __init__(self, in_ch=6, d=128):
        super().__init__()

        def blk(cin, cout):
            return torch.nn.Sequential(
                torch.nn.Conv2d(cin, cout, 4, stride=2, padding=1),
                torch.nn.GroupNorm(8, cout), torch.nn.SiLU())

        self.trunk = torch.nn.Sequential(
            blk(in_ch, 32), blk(32, 64), blk(64, 128),
            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
            torch.nn.GroupNorm(8, 256), torch.nn.SiLU(),
            torch.nn.Conv2d(256, 256, 3, stride=2, padding=1),
            torch.nn.GroupNorm(8, 256), torch.nn.SiLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(1024, 256), torch.nn.LayerNorm(256),
            torch.nn.SiLU(), torch.nn.Linear(256, d))
        self.d = d
        self.bounds = np.cumsum([0] + BLOCK_SIZES).tolist()
        self.lns = torch.nn.ModuleList(
            [torch.nn.LayerNorm(s) for s in BLOCK_SIZES])

    def forward(self, x):
        h = self.trunk(x)
        return torch.cat([ln(h[:, self.bounds[i]:self.bounds[i + 1]])
                          for i, ln in enumerate(self.lns)], dim=1)


class BlockHead(torch.nn.Module):
    """Predicts block k's own future at its own stride. The action
    summary is coarsened on purpose (horizon mean/last + scalars):
    coarse blocks must only see time-averaged action info (thermometer
    theorem)."""

    def __init__(self, d_in, a_dim=6, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in + a_dim, hidden), torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(), torch.nn.Linear(hidden, d_in))

    def forward(self, z_k, a_sum):
        return self.net(torch.cat([z_k, a_sum], dim=-1))


def block_slices():
    b = np.cumsum([0] + BLOCK_SIZES)
    return [slice(int(b[i]), int(b[i + 1])) for i in range(N_BLOCKS)]


def sample_batch(starts, ends, rng):
    """Shared start t; per-block horizon h_k = stride_k * random mult."""
    max_h = max(CUR_STRIDES) * max(MULTS)
    idxs, hs = [], [[] for _ in range(N_BLOCKS)]
    while len(idxs) < BATCH:
        e = int(rng.integers(len(starts)))
        s, en = int(starts[e]), int(ends[e])
        if en - s < max_h + 2:
            continue
        t = int(rng.integers(s, en - max_h - 1))
        idxs.append(t)
        for k in range(N_BLOCKS):
            hs[k].append(CUR_STRIDES[k]
                         * int(rng.integers(1, len(MULTS) + 1)))
    return np.asarray(idxs), [np.asarray(h) for h in hs]


def to_tensor(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device)


def make_batch(obs, actions, starts, ends, idxs, hs, device):
    """Online stacks, per-block target stacks, per-block action summaries.

    Horizons differ per sample within a block, so action statistics are
    computed per sample (no equal-length stacking).
    """
    B = len(idxs)
    ep_of = np.searchsorted(ends, idxs, side="right") - 1
    st_of = starts[ep_of]
    prev = np.maximum(idxs - 1, st_of)

    def stacks(frames):
        return np.stack([np.concatenate([obs[p], obs[i]], axis=-1)
                         for p, i in frames])

    xs = stacks(zip(prev, idxs))
    xs = to_tensor(xs / 255.0, device).permute(0, 3, 1, 2)
    xs = xs.reshape(B, -1, 56, 56)
    ys, a_sums = [], []
    for k in range(N_BLOCKS):
        h = hs[k]
        ti = idxs + h
        tp = np.maximum(ti - 1, st_of)
        y = to_tensor(stacks(zip(tp, ti)) / 255.0, device)
        ys.append(y.permute(0, 3, 1, 2).reshape(B, -1, 56, 56))
        a_sum = np.empty((B, 6), dtype=np.float32)
        for j, (i, hj) in enumerate(zip(idxs, h)):
            seg = actions[i:i + int(hj)] / 512.0
            a_sum[j] = [seg[:, 0].mean(), seg[:, 1].mean(),
                        seg[-1, 0], seg[-1, 1],
                        hj / (5 * 4), CUR_STRIDES[k] / 5.0]
        a_sums.append(to_tensor(a_sum, device))
    return xs, ys, a_sums


def train_arm(arm, strides, obs, actions, starts, ends, device, log):
    """One MRBP training run."""
    global CUR_STRIDES
    CUR_STRIDES = strides
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    enc = GroupedEncoder().to(device)
    tgt_enc = make_target(enc).to(device)
    heads = torch.nn.ModuleList(
        [BlockHead(s.stop - s.start) for s in block_slices()]).to(device)
    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(heads.parameters()), lr=3e-4)
    slices = block_slices()
    step, t0 = 0, time.time()
    enc.train()
    while step < STEPS:
        idxs, hs = sample_batch(starts, ends, rng)
        xs, ys, a_sums = make_batch(obs, actions, starts, ends,
                                    idxs, hs, device)
        z = enc(xs)
        with torch.no_grad():
            zt = [tgt_enc(y) for y in ys]
        T = temperature(step, STEPS)
        loss = 0.0
        rec = {"arm": arm, "stage": "wm", "step": step, "T": round(T, 4),
               "u": [], "s": [], "std": []}
        for k in range(N_BLOCKS):
            sl = slices[k]
            zk = z[:, sl]
            pred = heads[k](zk, a_sums[k])
            u = F.mse_loss(pred, zt[k][:, sl])
            s = gaussian_entropy(zk.cpu()).to(device)
            loss = loss + (u - T * s) / N_BLOCKS
            rec["u"].append(round(u.item(), 4))
            rec["s"].append(round(s.item(), 3))
            rec["std"].append(round(zk.std().item(), 3))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        emma_update(enc, tgt_enc, momentum=0.999)
        if step % 250 == 0:
            rec["sec"] = round(time.time() - t0, 1)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[{arm}] step {step}: u={rec['u']} s={rec['s']} "
                  f"std={rec['std']}", flush=True)
        step += 1
    out_dir = os.path.join("results", arm)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"enc": enc.state_dict(), "heads": heads.state_dict(),
                "strides": strides},
               os.path.join(out_dir, "world_model.pt"))
    return enc


def measure_arm(enc, strides, obs, starts, ends, device):
    """Per-block tau / probes / std (same estimators as every audit)."""
    enc.eval()
    z_eps = encode_episodes(enc, obs, starts, ends, device)
    tau, _ = tau_acf(z_eps)
    slices = block_slices()
    Z = np.concatenate(z_eps)
    dnp = np.load(PROBE_NPZ)
    n = 4000
    Zp = encode_batch(enc, dnp["imgs"][:n], device)
    out = {"blocks": []}
    for k, sl in enumerate(slices):
        idx = np.arange(sl.start, sl.stop)
        out["blocks"].append({
            "stride": strides[k],
            "tau_median": float(np.median(tau[idx])),
            "block_r2": float(ridge_r2(Zp[:, idx], dnp["block"][:n]).mean()),
            "agent_r2": float(ridge_r2(Zp[:, idx], dnp["agent"][:n]).mean()),
            "std": float(Z[:, idx].std(axis=0).mean()),
        })
    return out


def score(entry):
    b = entry["blocks"]
    taus = [x["tau_median"] for x in b]
    ordered = taus[0] >= taus[1] >= taus[2]
    ratio = taus[0] / max(taus[2], 1e-9)
    c1 = bool(ordered and ratio >= TAU_RATIO)
    c2 = bool(b[0]["block_r2"] - b[2]["block_r2"] >= R2_GAP
              and b[2]["agent_r2"] - b[0]["agent_r2"] >= 0.05)
    c3 = bool(all(x["std"] >= 0.01 for x in b))
    return {"c1_tau_clusters": c1, "c2_content_placement": c2,
            "c3_survival": c3,
            "tau_ratio": round(float(ratio), 2),
            "block_taus": [round(t, 1) for t in taus],
            "pass_all": bool(c1 and c2 and c3)}


def main():
    device = device_auto()
    obs, actions, _, starts, ends, _ = load_cache()
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    log = open("results/mrbp_metrics.jsonl", "a")
    for arm, strides in [("mrbp_v3", STRIDES_V3),
                         ("mrbp_single", [1, 1, 1])]:
        if arm in results and "criteria" in results.get(arm, {}):
            print(f"[{arm}] already done, skipping", flush=True)
            continue
        print(f"[{arm}] training {STEPS} steps, strides={strides} ...",
              flush=True)
        enc = train_arm(arm, strides, obs, actions, starts, ends,
                        device, log)
        entry = measure_arm(enc, strides, obs, starts, ends, device)
        entry["criteria"] = score(entry)
        results[arm] = entry
        json.dump(results, open(OUT, "w"), indent=2)
        c = entry["criteria"]
        print(f"[{arm}] taus={c['block_taus']} ratio={c['tau_ratio']} "
              f"criteria={c}", flush=True)
    log.close()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
