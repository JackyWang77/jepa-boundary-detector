"""Remap raw HF LeWM checkpoint to the key layout stable_worldmodel expects.

The HF checkpoints under ``quentinll/lewm-*`` store the ViT encoder with the
legacy transformers naming (``encoder.encoder.layer.N.attention.attention.query``)
while the current ``vit_hf`` builds ``encoder.layers.N.attention.q_proj``. This
rewrites ``weights.pt`` in-place so ``swm.wm.utils.load_pretrained('pusht/weights.pt')``
(used by e6_eval.py) loads cleanly on the 4060 machine.

Usage (STABLEWM_HOME points at the checkpoint root):
    STABLEWM_HOME=D:/stable_worldmodel python convert_weights_pt.py --env pusht
"""
import argparse
import re
import os
from pathlib import Path

import torch


def remap_key(k: str) -> str:
    """Map raw HF-style key to the timm-style key current vit_hf expects."""
    m = re.match(r'^encoder\.encoder\.layer\.(\d+)\.(.*)$', k)
    if not m:
        return k
    layer, rest = m.group(1), m.group(2)
    rules = [
        ('attention.attention.query', 'attention.q_proj'),
        ('attention.attention.key', 'attention.k_proj'),
        ('attention.attention.value', 'attention.v_proj'),
        ('attention.output.dense', 'attention.o_proj'),
        ('intermediate.dense', 'mlp.fc1'),
        ('output.dense', 'mlp.fc2'),
    ]
    for old, new in rules:
        if rest.startswith(old):
            rest = new + rest[len(old):]
            break
    return f'encoder.layers.{layer}.{rest}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='pusht')
    args = ap.parse_args()

    home = Path(os.environ.get('STABLEWM_HOME', Path.home() / '.stable_worldmodel'))
    ckpt_dir = home / 'checkpoints' / args.env
    weights = ckpt_dir / 'weights.pt'

    sd = torch.load(weights, map_location='cpu', weights_only=False)
    new_sd = {remap_key(k): v for k, v in sd.items()}

    n_remapped = sum(1 for k in sd if k != remap_key(k))
    # back up the raw HF file first
    backup = weights.with_suffix('.pt.hf-raw')
    if not backup.exists():
        backup.write_bytes(weights.read_bytes())

    # load the model the same way load_pretrained does, to verify keys strictly
    import json
    from hydra.utils import instantiate
    cfg = json.loads((ckpt_dir / 'config.json').read_text())
    model = instantiate(cfg)
    model.load_state_dict(new_sd, strict=True)

    torch.save(new_sd, weights)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'OK -> {weights} ({n_params:.1f}M params, '
          f'{len(new_sd)} tensors, {n_remapped} remapped)')


if __name__ == '__main__':
    main()