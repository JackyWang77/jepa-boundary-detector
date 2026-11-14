# R2 Cross-Capacity Experiment (reviewer response)

Goal: answer reviewer concern 2 with a real experiment. Run the same flat grid and
probe schedule against a **smaller-capacity frozen LeWM trained on PushT**, and
compare with the main (published) model. All work happens on the 4060 (conda `aiim`),
mac only consumes the JSONs.

## Phase 0 — train the small-capacity LeWM (4060, ~hours)

Capacity knobs that do NOT touch the encoder (ViT-tiny embed dim 192 is fixed):

| knob | main model | small model |
| --- | --- | --- |
| predictor depth | 6 | **3** |
| predictor mlp_dim | 2048 | **1024** |

Why halve the predictor instead of swapping the encoder for ViT-Small:
changing the encoder would move the latent dim from 192 to 384, which breaks
the detector head input, the policy, and the CEM latent-distance code — a
half-day budget cannot absorb that, and it would add a second variable to the
comparison. Halving the predictor keeps the entire downstream stack untouched
and still answers "different-capacity JEPA" head-on.

1. `cd aiim2026/third_party/le-wm`
2. Copy `config/train/lewm.yaml` to `config/train/lewm_small.yaml`, set
   `model.predictor.depth: 3`, `model.predictor.mlp_dim: 1024`, keep data=pusht.
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

## Phase 4 — Cube detector-layer evidence (no training, minutes)

Reviewer R1 asks for signal distributions per task type plus a false-fire
analysis. This is independent of whether the probe schedule wins on Cube — the
detector evidence stands alone. All assets already exist on the 4060
(`features_cube/` npz files and the trained cube head in `models/`).

1. Write and run `aiim2026/src/r1_cube_hist_tau.py` (conda `aiim`, CPU-only):
   a. Load `features_cube/ep*.npz` + `features_cube/meta.json` + the deployed
      cube head weights; replicate the mac-side event-vs-non-event histograms
      of `aiim2026/src/r1_review_analysis.py` Part A (same colors #0072BD /
      #8C8C8C, same 3-panel layout, titles f1 energy / f2 cosine / f3 novelty)
      -> `aiim2026/results/fig3_cube_dist.pdf`
   b. Cube hold-out tau sweep, tau in {0.30, ..., 0.70} step 0.05
      (refractory=5, match tol=2) -> precision / recall / F1 /
      false_fires_per_episode per tau
      -> `aiim2026/results/r1_cube_hist_tau.json`
2. JSON schema must mirror the mac-side `aiim2026/results/r1_review.json`
   (`part_d_tau_sweep_pushT` entries) so the mac consumes it directly.
3. PROGRESS.md entry + git push.

## Done criteria

- JSONs in `aiim2026/runs/`, one PROGRESS.md line
  (`日期 | [4060 agent] **pusht_small** | 数据 | 判读`), push to git.
- Mac then folds the numbers into Results/Conclusion and recompiles.
