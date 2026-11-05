"""Download pusht dataset + cube checkpoint via hf-mirror (resume + retry).

Key fixes vs the earlier stuck download:
  * HF_HUB_DISABLE_XET=1 -> bypass the slow cas-bridge.xethub (Xet LFS)
    backend and use the legacy LFS CDN (mirrored by hf-mirror).
  * pusht dataset lives in the *dataset* repo of quentinll/lewm-pusht
    (repo_type='dataset'), NOT a model repo.
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['HF_HUB_DISABLE_XET'] = '1'

from huggingface_hub import hf_hub_download

ROOT = Path(os.environ.get('STABLEWM_HOME', Path.home() / '.stable_worldmodel'))


def _download(repo_id, filename, repo_type, local_dir, tag):
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f'[{tag}] start {repo_id}/{filename} -> {local_dir}', flush=True)
    t0 = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            p = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                local_dir=str(local_dir),
                resume_download=True,
            )
            dt = time.time() - t0
            size = Path(p).stat().st_size
            print(f'[{tag}] DONE attempt={attempt} {size} bytes '
                  f'{size / 1e6:.1f}MB in {dt:.0f}s '
                  f'({size / 1e6 / dt:.2f} MB/s)', flush=True)
            return p
        except Exception as e:
            print(f'[{tag}] FAIL attempt={attempt}: '
                  f'{type(e).__name__} {e}', flush=True)
            time.sleep(30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pusht', action='store_true', help='download pusht dataset')
    ap.add_argument('--cube', action='store_true', help='download cube checkpoint')
    ap.add_argument('--all', action='store_true', help='download both')
    args = ap.parse_args()

    if args.pusht or args.all:
        _download(
            'quentinll/lewm-pusht', 'pusht_expert_train.h5.zst',
            'dataset', ROOT / 'datasets', 'pusht',
        )
    if args.cube or args.all:
        for fn in ('config.json', 'weights.pt'):
            _download(
                'quentinll/lewm-cube', fn,
                'model', ROOT / 'checkpoints' / 'cube', 'cube',
            )


if __name__ == '__main__':
    main()