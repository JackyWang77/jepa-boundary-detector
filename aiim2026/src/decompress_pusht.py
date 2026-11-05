"""Decompress pusht_expert_train.h5.zst -> .h5 via python-zstandard (streaming)."""
import time
from pathlib import Path

import zstandard as zstd

SRC = Path(r'D:\stable_worldmodel\datasets\pusht_expert_train.h5.zst')
DST = Path(r'D:\stable_worldmodel\datasets\pusht_expert_train.h5')

t0 = time.time()
dctx = zstd.ZstdDecompressor()
with open(SRC, 'rb') as src, open(DST, 'wb') as dst:
    dctx.copy_stream(src, dst)
size = DST.stat().st_size
print(f'DONE size={size} ({size/1e9:.2f} GB) in {(time.time()-t0)/60:.1f} min',
      flush=True)
