"""Sequential ranged download of the pusht dataset (bypasses hf_hub's Xet cache).

hf_hub's Xet backend tries to create a `.cache` dir under the dataset dir and
gets blocked by AV (WinError 5). Plain requests streaming avoids that. The
signed CDN URL (~us.aws.cdn.hf.co) expires after ~1h, so we re-resolve and
resume from the current byte offset on 401/403.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import requests

HF_MIRROR = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
RESOLVE = f'{HF_MIRROR}/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst'
EXPECTED = 13_136_247_974


def resolve(session):
    r = session.get(RESOLVE, stream=True, allow_redirects=True, timeout=60)
    r.close()
    return r.url


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default=r'D:\stable_worldmodel\datasets\pusht_expert_train.h5.zst')
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    pos = dest.stat().st_size if dest.exists() else 0
    print(f'resume from {pos / 1e9:.3f} GB / {EXPECTED / 1e9:.2f} GB', flush=True)

    last_print = time.time()
    last_pos = pos
    t_start = time.time()

    while pos < EXPECTED:
        try:
            url = resolve(session)
            headers = {'Range': f'bytes={pos}-'} if pos > 0 else {}
            with session.get(url, headers=headers, stream=True, timeout=120) as r:
                if r.status_code not in (200, 206):
                    print(f'  status {r.status_code}; re-resolving after 5s', flush=True)
                    time.sleep(5)
                    continue
                with open(dest, 'ab') as f:
                    for chunk in r.iter_content(8 * 1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pos += len(chunk)
                        now = time.time()
                        if now - last_print >= 15:
                            rate = (pos - last_pos) / (now - last_print) / 1e6
                            eta = (EXPECTED - pos) / 1e6 / rate if rate > 0 else 0
                            print(f'  {pos / 1e9:.2f}GB/{EXPECTED / 1e9:.2f}GB '
                                  f'({100 * pos / EXPECTED:.1f}%) {rate:.2f} MB/s '
                                  f'ETA {eta / 60:.0f}min', flush=True)
                            last_print = now
                            last_pos = pos
        except Exception as e:
            print(f'  ERR {type(e).__name__}: {e}; retry in 5s (pos={pos})', flush=True)
            time.sleep(5)

    dt = time.time() - t_start
    got = dest.stat().st_size
    ok = got == EXPECTED
    print(f'DONE size={got} expected={EXPECTED} match={ok} '
          f'in {dt / 60:.1f}min ({got / 1e6 / dt:.2f} MB/s avg)', flush=True)


if __name__ == '__main__':
    main()