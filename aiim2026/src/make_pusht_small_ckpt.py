"""Materialize $STABLEWM_HOME/checkpoints/pusht_small/weights.pt from a training
epoch checkpoint, and fix the config targets to the stable-worldmodel package.

The trainer rewrites config.json every epoch with the le-wm repo's own module
paths (jepa.JEPA), which the plan scripts cannot import. This script rewrites
them and verifies the state dict loads strictly into LeWM.

  python src/make_pusht_small_ckpt.py            # newest weights_epoch_N.pt
  python src/make_pusht_small_ckpt.py 12         # a specific epoch
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate

CKPT = Path(os.environ.get('STABLEWM_HOME', r'D:\stable_worldmodel')) \
    / 'checkpoints' / 'pusht_small'
TARGETS = {
    '': 'stable_worldmodel.wm.lewm.LeWM',
    'predictor': 'stable_worldmodel.wm.lewm.module.Predictor',
    'action_encoder': 'stable_worldmodel.wm.lewm.module.Embedder',
    'projector': 'stable_worldmodel.wm.lewm.module.MLP',
    'pred_proj': 'stable_worldmodel.wm.lewm.module.MLP',
}


def main():
    epochs = [int(re.search(r'_(\d+)\.pt$', p.name).group(1))
              for p in CKPT.glob('weights_epoch_*.pt')]
    epoch = int(sys.argv[1]) if len(sys.argv) > 1 else max(epochs)
    src = CKPT / f'weights_epoch_{epoch}.pt'
    if not src.exists():
        sys.exit(f'missing {src}')

    cfg = json.loads((CKPT / 'config.json').read_text())
    cfg['_target_'] = TARGETS['']
    for key, target in TARGETS.items():
        if key:
            cfg[key]['_target_'] = target
    (CKPT / 'config.json').write_text(json.dumps(cfg, indent=4))

    model = instantiate(cfg)
    model.load_state_dict(torch.load(src, map_location='cpu'), strict=True)
    shutil.copyfile(src, CKPT / 'weights.pt')
    n = sum(p.numel() for p in model.parameters())
    print(f'epoch {epoch}: strict load OK, {n / 1e6:.2f}M params, '
          f'predictor depth {cfg["predictor"]["depth"]} '
          f'mlp {cfg["predictor"]["mlp_dim"]}')


if __name__ == '__main__':
    main()
