# R2 Cross-Capacity Experiment (reviewer response)

Goal: answer reviewer concern 2 with a real experiment. Run the same flat grid and
probe schedule against a **smaller-capacity frozen LeWM trained on PushT**, and
compare with the main (published) model. All work happens on the 4060 (conda `aiim`),
mac only consumes the JSONs.

## Phase 0 — train the small-capacity LeWM (4060, ~hours)

Capacity knobs that do NOT touch the encoder (ViT-tiny embed dim 192 is fixed):

| knob | main model | small model |
| --- | --- | --- |
| predictor depth | 6 | **2** |
| predictor mlp_dim | 2048 | **1024** |

1. `cd aiim2026/third_party/le-wm`
2. Copy `config/train/lewm.yaml` to `config/train/lewm_small.yaml`, set
   `model.predictor.depth: 2`, `model.predictor.mlp_dim: 1024`, keep data=pusht.
3. Launch training per le-wm README (`python train.py --config-name=...`; confirm the
   entrypoint with `train.py --help` first). Record the epoch count and wall time.

## Phase 1 — convert to stable-worldmodel format

Reuse the conversion used for the main model (see EXPERIMENT_GUIDE §2 / handshake §2:
`quentinll/lewm-pusht` weights.pt → `checkpoints/pusht/{config.json,weights.pt}`),
with the new checkpoint. Output: `checkpoints/pusht_small/{config.json,weights.pt}`.
`torch.load` verify before and after, as always.

## Phase 2 — flat grid + zero-shot probe (existing head, no retraining)

Command template = handshake §4, only the policy path changes
(`policy=pusht_small/weights.pt`). cwd=`aiim2026/third_party/stable-worldmodel/scripts/plan`.

- flat: `h∈{5,10,15}` × `d∈{50,75}` × seed 42/43/44 × `eval.num_eval=150` (18 runs)
- probe: same grid + `+arm=probe_t` (6 runs)
- out_json to `aiim2026/runs/pusht_small_{arm}_d{d}_s{seed}.json`

Interpretation (pre-registered, no after-the-fact tuning):
- probe significantly beats the best flat at both distances → the head transfers
  across capacity with zero retraining, strongest possible answer.
- probe loses zero-shot → proceed to Phase 3.
- probe loses after Phase 3 as well → report honestly as a limitation.

## Phase 3 — (conditional) re-distill the head on the small model

1. `e2_extract_features.py` with the small checkpoint → `features_small/`
2. `r2_dualhead.py` on those features (same event labels, gates untouched) → small head
3. re-run the 6 probe runs with the new head

## Done criteria

- JSONs in `aiim2026/runs/`, one PROGRESS.md line
  (`日期 | [4060 agent] **pusht_small** | 数据 | 判读`), push to git.
- Mac then folds the numbers into Results/Conclusion and recompiles.
