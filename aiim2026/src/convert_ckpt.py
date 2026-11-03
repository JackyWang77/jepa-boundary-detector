"""Convert HF LeWM checkpoint (weights.pt + config.json) to _object.ckpt for swm eval.

Remaps HF-transformers ViT naming inside the checkpoint to the timm-style naming
expected by stable_worldmodel's LeWM implementation, then saves the instantiated
model object as required by eval.py / AutoCostModel.

Usage:
    python convert_ckpt.py [--env pusht]
"""
import argparse
import json
import re
from pathlib import Path

import hydra
import torch


def remap_key(k: str) -> str:
    """Map checkpoint (HF-style) key to model (timm-style) key."""
    m = re.match(r"^encoder\.encoder\.layer\.(\d+)\.(.*)$", k)
    if not m:
        return k
    layer, rest = m.group(1), m.group(2)
    rules = [
        ("attention.attention.query", "attention.q_proj"),
        ("attention.attention.key", "attention.k_proj"),
        ("attention.attention.value", "attention.v_proj"),
        ("attention.output.dense", "attention.o_proj"),
        ("intermediate.dense", "mlp.fc1"),
        ("output.dense", "mlp.fc2"),
    ]
    for old, new in rules:
        if rest.startswith(old):
            rest = new + rest[len(old):]
            break
    return f"encoder.layers.{layer}.{rest}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="pusht")
    args = ap.parse_args()

    src = Path.home() / ".stable-wm" / "pusht_hf"
    out = Path.home() / ".stable-wm" / "pusht" / "lewm_object.ckpt"
    out.parent.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    model = hydra.utils.instantiate(cfg)
    sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)

    new_sd = {remap_key(k): v for k, v in sd.items()}
    model_keys = set(model.state_dict().keys())
    remapped = set(new_sd.keys())

    missing = model_keys - remapped
    unexpected = remapped - model_keys
    if missing or unexpected:
        print(f"missing ({len(missing)}):", sorted(missing)[:8])
        print(f"unexpected ({len(unexpected)}):", sorted(unexpected)[:8])
        raise SystemExit("key remap incomplete, fix rules above")

    model.load_state_dict(new_sd, strict=True)
    torch.save(model, out)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"OK -> {out} ({n_params:.1f}M params, {len(new_sd)} tensors)")


if __name__ == "__main__":
    main()
