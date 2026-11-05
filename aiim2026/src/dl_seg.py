"""Segmented parallel download of pusht dataset via hf-mirror (VPN off, direct CN).

Splits the 12.23GB .zst into 16 segments, 4 concurrent workers, each segment
resumable (append mode + Range). Final step: concatenate + size check.
"""
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

URL = 'https://hf-mirror.com/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst'
TOTAL = 13_136_247_974
NSEG = 16
WORKERS = 4
PARTS = Path(r'D:\stable_worldmodel\datasets\pusht_parts')
FINAL = Path(r'D:\stable_worldmodel\datasets\pusht_expert_train.h5.zst')


def seg_range(i: int):
    lo = TOTAL * i // NSEG
    hi = TOTAL * (i + 1) // NSEG - 1
    return lo, hi


def fetch_seg(i: int) -> int:
    lo, hi = seg_range(i)
    seg_len = hi - lo + 1
    p = PARTS / f'seg_{i:02d}.bin'
    pos = p.stat().st_size if p.exists() else 0
    if pos >= seg_len:
        print(f'seg{i:02d}: already done ({pos})', flush=True)
        return pos
    while pos < seg_len:
        try:
            headers = {'Range': f'bytes={lo + pos}-{hi}'}
            with requests.Session() as s:
                with s.get(URL, headers=headers, stream=True,
                           timeout=(30, 120)) as r:
                    if r.status_code not in (200, 206):
                        print(f'seg{i:02d}: status {r.status_code}, retry 5s',
                              flush=True)
                        time.sleep(5)
                        continue
                    with open(p, 'ab') as f:
                        for chunk in r.iter_content(4 * 1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                pos += len(chunk)
        except Exception as e:
            print(f'seg{i:02d}: ERR {type(e).__name__}: {e} at pos={pos}',
                  flush=True)
            time.sleep(5)
    print(f'seg{i:02d}: complete ({pos})', flush=True)
    return pos


def progress_loop(done_event, t_start):
    while not done_event.is_set():
        time.sleep(30)
        total_done = 0
        for i in range(NSEG):
            p = PARTS / f'seg_{i:02d}.bin'
            total_done += min(p.stat().st_size if p.exists() else 0,
                              seg_range(i)[1] - seg_range(i)[0] + 1)
        dt = time.time() - t_start
        rate = total_done / dt / 1e6 if dt > 0 else 0
        eta = (TOTAL - total_done) / 1e6 / rate / 60 if rate > 0 else 0
        print(f'[progress] {total_done/1e9:.2f}/{TOTAL/1e9:.2f} GB '
              f'({100*total_done/TOTAL:.1f}%) avg {rate:.2f} MB/s '
              f'ETA {eta:.0f} min', flush=True)


def main():
    import threading
    PARTS.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    done = threading.Event()
    pt = threading.Thread(target=progress_loop, args=(done, t_start),
                          daemon=True)
    pt.start()
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(fetch_seg, range(NSEG)))
    done.set()

    # concatenate
    print('all segments done, concatenating...', flush=True)
    if FINAL.exists():
        FINAL.unlink()
    with open(FINAL, 'wb') as out:
        for i in range(NSEG):
            p = PARTS / f'seg_{i:02d}.bin'
            with open(p, 'rb') as f:
                shutil.copyfileobj(f, out, 64 * 1024 * 1024)
    size = FINAL.stat().st_size
    ok = size == TOTAL
    print(f'FINAL size={size} expected={TOTAL} match={ok}', flush=True)
    if ok:
        shutil.rmtree(PARTS)
        print('parts cleaned. DONE.', flush=True)
    else:
        print('SIZE MISMATCH — parts kept for resume.', flush=True)


if __name__ == '__main__':
    main()
