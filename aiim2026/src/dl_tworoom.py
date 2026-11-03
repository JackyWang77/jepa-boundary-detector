"""Download tworoom dataset via huggingface_hub (resume + retry + heartbeat)."""
import os
import subprocess
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import hf_hub_download

DEST = '/Users/wanglanlan/course/Quentropy/E2_JEPA/aiim2026/external/lewm-tworoom'
CACHE = os.path.expanduser('~/.cache/huggingface')

def cache_size():
    r = subprocess.run(['du', '-sh', CACHE], capture_output=True, text=True)
    return r.stdout.split()[0] if r.stdout else '?'

attempt = 0
while True:
    attempt += 1
    print(f'[heartbeat] attempt={attempt} cache={cache_size()}',
          flush=True)
    try:
        p = hf_hub_download(
            repo_id='quentinll/lewm-tworooms',
            filename='tworoom.tar.zst',
            local_dir=DEST,
            resume_download=True,
            local_files_only=False,
        )
        print(f'DONE attempt={attempt} -> {p}', flush=True)
        break
    except Exception as e:
        print(f'FAIL attempt={attempt}: {type(e).__name__} {e}',
              flush=True)
        time.sleep(30)