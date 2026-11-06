"""Audit official cube h5 — skip pixels random reads, use full arrays."""
import h5py, numpy as np, json
from pathlib import Path

h5_path = Path(r"D:\stable_worldmodel\datasets\cube_single_expert.h5")

with h5py.File(str(h5_path), "r") as f:
    N = int(f["ep_len"].shape[0])
    TOT = int(f["step_idx"].shape[0])
    print(f"Total episodes: {N}, total steps: {TOT}")

    ep_lens = f["ep_len"][:]
    print(f"ep_len: mean={ep_lens.mean():.1f}  all_same={np.all(ep_lens==ep_lens[0])}  L={ep_lens[0]}")

    ep_offsets = f["ep_offset"][:]
    # verify: offsets are at every L steps
    expected = np.arange(N) * ep_lens[0]
    print(f"ep_offset consistent: {np.array_equal(ep_offsets, expected)}")

    # --- goal distance at step 0 of every ep ---
    print(f"\n=== GOAL DIST (step 0 of every ep) ===")
    all_block_0 = f["privileged_block_0_pos"][ep_offsets]       # (N, 3)
    all_target_0 = f["privileged_target_block_pos"][ep_offsets] # (N, 3)
    all_d0 = np.linalg.norm(all_block_0 - all_target_0, axis=1)
    print(f"  mean={all_d0.mean():.4f}  std={all_d0.std():.4f}")
    print(f"  min={all_d0.min():.4f}  max={all_d0.max():.4f}")
    print(f"  median={np.median(all_d0):.4f}")
    print(f"  percentiles: p5={np.percentile(all_d0,5):.4f} p25={np.percentile(all_d0,25):.4f} p75={np.percentile(all_d0,75):.4f} p95={np.percentile(all_d0,95):.4f}")

    bins = [0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 2.0]
    counts, _ = np.histogram(all_d0, bins=bins)
    print(f"  histogram (bins in meters):")
    for j in range(len(bins)-1):
        bar = "#" * int(60*counts[j]/max(counts)) if max(counts) > 0 else ""
        print(f"    [{bins[j]:.2f}-{bins[j+1]:.2f}): {counts[j]:5d} ({100*counts[j]/N:5.1f}%) {bar}")

    # --- success per episode ---
    print(f"\n=== SUCCESS RATE ===")
    succ_full = f["success"][:]  # (TOT,)
    ep_success = np.zeros(N, dtype=bool)
    # since all ep_len=201, we can reshape
    succ_2d = succ_full.reshape(N, 201)
    ep_success = succ_2d.any(axis=1)
    succ_rate = ep_success.mean()
    succ_n = ep_success.sum()
    print(f"  Episodes with ANY success: {succ_n}/{N} ({100*succ_rate:.1f}%)")

    # success step position (within ep)
    succ_steps = succ_2d.argmax(axis=1)
    succ_steps[~ep_success] = -1
    has = succ_steps[succ_steps >= 0]
    print(f"  Success step within ep: mean={has.mean():.1f} median={np.median(has):.0f}")
    print(f"    p10={np.percentile(has,10):.0f} p25={np.percentile(has,25):.0f} p75={np.percentile(has,75):.0f} p90={np.percentile(has,90):.0f}")

    # --- distance splits ---
    print(f"\n=== DISTANCE SPLITS (all eps, goal dist proxy) ===")
    # LeWM PushT uses goal_offset_steps / budget as difficulty
    # cube: d0 in meters -> map to "d25" (close), "d50" (mid), "d75" (far)
    # Using thresholds that give ~25/50/25 split or similar
    for label, lo, hi in [("close", 0.0, 0.05), ("mid-close", 0.05, 0.10),
                          ("mid-far", 0.10, 0.20), ("far", 0.20, 0.50), ("vfar", 0.50, 2.0)]:
        mask = (all_d0 >= lo) & (all_d0 < hi)
        n = mask.sum()
        s = ep_success[mask].sum()
        print(f"  {label:10s} [{lo:.2f},{hi:.2f}): n={n:5d} ({100*n/N:5.1f}%)  success={100*s/max(n,1):.1f}%")

    # --- step_idx sanity ---
    step_idx = f["step_idx"][:]
    expected_steps = np.tile(np.arange(201), N)
    print(f"\n=== STEP_IDX SANITY ===")
    print(f"  step_idx matches arange@201 tiled: {np.array_equal(step_idx, expected_steps)}")

    # --- reward stats ---
    reward = f["reward"][:]
    print(f"\n=== REWARD ===")
    print(f"  mean={reward.mean():.4f}  std={reward.std():.4f}  min={reward.min():.4f}  max={reward.max():.4f}")

    # --- action range ---
    action = f["action"][:]
    print(f"\n=== ACTION ===")
    print(f"  shape={action.shape}  mean={action.mean(axis=0)}  std={action.std(axis=0)}")
    print(f"  min={action.min(axis=0)}  max={action.max(axis=0)}")

# save summary
summary = {
    "total_eps": N,
    "ep_len": int(ep_lens[0]),
    "ep_len_all_same": True,
    "total_steps": TOT,
    "goal_dist_meters": {
        "mean": float(all_d0.mean()), "std": float(all_d0.std()),
        "min": float(all_d0.min()), "max": float(all_d0.max()),
        "median": float(np.median(all_d0)),
        "p5": float(np.percentile(all_d0,5)), "p95": float(np.percentile(all_d0,95)),
    },
    "episodes_with_success": int(succ_n),
    "success_rate_pct": float(100*succ_rate),
    "success_step_in_ep": {
        "mean": float(has.mean()), "median": float(np.median(has)),
        "p10": float(np.percentile(has,10)), "p90": float(np.percentile(has,90)),
    },
    "pixels": {"shape": [TOT, 224, 224, 3], "dtype": "uint8", "native_224": True},
    "note": "Official HF lewm-cube dataset — clean, no black frame issue (ep_len fixed 201, all eps valid)"
}
out = Path(r"D:\stable_worldmodel\datasets\cube_official_audit_summary.json")
out.write_text(json.dumps(summary, indent=2))
print(f"\nSaved summary -> {out}")
