"""Extract cube tar.zst -> tar -> h5, then audit."""
import tarfile, time, os, sys, json, h5py, numpy as np
import zstandard as zstd
from pathlib import Path

TAR_ZST = Path(r"D:\stable_worldmodel\datasets\cube_single_expert.tar.zst")
TAR_RAW = Path(r"D:\stable_worldmodel\datasets\cube_single_expert.tar")
OUT = Path(r"D:\stable_worldmodel\datasets")

# --- 1. decompress zst -> tar ---
if not TAR_RAW.exists():
    t0 = time.time()
    print(f"[zst->tar] decompressing {TAR_ZST.stat().st_size/1e9:.2f}GB...")
    dctx = zstd.ZstdDecompressor()
    with open(TAR_ZST, "rb") as fin, open(TAR_RAW, "wb") as fout:
        dctx.copy_stream(fin, fout)
    print(f"[zst->tar] DONE in {time.time()-t0:.1f}s  size={TAR_RAW.stat().st_size/1e9:.2f}GB")
else:
    print(f"[zst->tar] already exists: {TAR_RAW.stat().st_size/1e9:.2f}GB")

# --- 2. extract tar ---
h5_path = OUT / "cube_single_expert.h5"
if not h5_path.exists():
    t0 = time.time()
    print(f"[tar->h5] extracting...")
    with tarfile.open(str(TAR_RAW), "r") as tar:
        members = tar.getmembers()
        print(f"[tar->h5] {len(members)} members:")
        for m in members:
            print(f"  {m.size/1e9:.2f}GB  {m.name}")
        tar.extractall(str(OUT))
    print(f"[tar->h5] DONE in {time.time()-t0:.1f}s")
else:
    print(f"[tar->h5] already exists: {h5_path.stat().st_size/1e9:.2f}GB")

# --- 3. audit ---
print(f"\n{'='*60}")
print("AUDIT OFFICIAL cube_single_expert.h5")
print(f"{'='*60}")

with h5py.File(str(h5_path), "r") as f:
    print(f"\nTop-level keys: {list(f.keys())}")
    for k in f.keys():
        obj = f[k]
        if isinstance(obj, h5py.Group):
            print(f"  GROUP {k}: keys={list(obj.keys())[:15]}")
        else:
            print(f"  DATASET {k}: shape={obj.shape} dtype={obj.dtype}")

    # episodes
    eps_grp = None
    for ek in ("episodes", "episode", "trajectories", "rollouts"):
        if ek in f:
            eps_grp = f[ek]
            break

    if eps_grp is None:
        # maybe flat dict-like
        print("\nTrying flat iteration...")
        ep_count = 0
        for k in f:
            ep_count += 1
            if ep_count <= 3:
                v = f[k]
                if hasattr(v, "shape"):
                    print(f"  {k}: shape={v.shape} dtype={v.dtype}")
                else:
                    print(f"  {k}: group with keys={list(v.keys())[:5]}")
        print(f"Total top-level entries: {ep_count}")
    else:
        ep_ids = list(eps_grp.keys())
        print(f"\nEpisodes group: {len(ep_ids)} episodes")

        rng = np.random.RandomState(42)
        sample_ids = rng.choice(ep_ids, size=min(30, len(ep_ids)), replace=False)

        stats = {"ep_lens": [], "black_mid": [], "black_first": [], "has_success": []}
        keys_seen = set()

        for eid in sample_ids:
            ep = eps_grp[eid]
            if isinstance(ep, h5py.Group):
                keys = list(ep.keys())
            else:
                continue
            keys_seen.update(keys)

            # frames / pixels
            frames = None
            for fk in ("frames", "pixels", "observations", "obs", "states", "image"):
                if fk in keys:
                    frames = ep[fk]
                    break

            if frames is not None and hasattr(frames, "shape") and len(frames.shape) >= 3:
                T = frames.shape[0]
                stats["ep_lens"].append(T)
                mid = frames[min(T//2, T-1)]
                first = frames[0]
                stats["black_mid"].append(bool(np.all(mid == 0) or (np.std(mid) < 0.5 and np.mean(mid) < 5)))
                stats["black_first"].append(bool(np.all(first == 0) or (np.std(first) < 0.5 and np.mean(first) < 5)))

            # success / reward
            for sk in ("success", "done", "rewards", "reward", "is_success"):
                if sk in keys:
                    vals = ep[sk][:]
                    stats["has_success"].append(float(np.max(vals) if vals.size else 0))
                    break

        print(f"\nKeys seen across sample: {sorted(keys_seen)}")
        print(f"\n{'='*60}")
        print("SUMMARY (n=30 sample)")
        print(f"{'='*60}")
        print(f"Total episodes in file: {len(ep_ids)}")
        if stats["ep_lens"]:
            el = np.array(stats["ep_lens"])
            print(f"Ep length: mean={el.mean():.0f}  std={el.std():.0f}  min={el.min()}  max={el.max()}")
            print(f"  median={np.median(el):.0f}")
        bm = sum(stats["black_mid"])
        bf = sum(stats["black_first"])
        print(f"Black mid-frame: {bm}/{len(stats['black_mid'])} ({100*bm/len(stats['black_mid']):.1f}%)")
        print(f"Black first-frame: {bf}/{len(stats['black_first'])} ({100*bf/len(stats['black_first']):.1f}%)")
        if stats["has_success"]:
            hs = np.array(stats["has_success"])
            print(f"Success flag present: {100*hs.mean():.1f}% eps have success signal")
