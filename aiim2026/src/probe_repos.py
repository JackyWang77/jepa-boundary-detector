"""Probe HF repo file lists (via hf-mirror) to pin exact dataset/checkpoint paths."""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import HfApi

api = HfApi()
for r in ['quentinll/lewm-pusht', 'quentinll/lewm-pusht-data', 'quentinll/lewm-cube']:
    print(f'== {r}')
    try:
        for f in api.list_repo_files(r):
            print('   ', f)
    except Exception as e:
        print('   ERR', type(e).__name__, e)