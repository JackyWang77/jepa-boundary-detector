# JEPA Boundary Detector

Latent boundary detection for planning and execution re-anchoring in frozen JEPA world models.

**Paper**: *Latent Boundary Detection for Planning and Execution Re-Anchoring in Frozen JEPA World Models*, AIIM 2026 (submitted).

## Idea

Rollouts of a frozen JEPA world model drift from the real dynamics, and the error compounds with every imagined step. Instead of retraining the model or planning with subgoals, this work reads the boundary structure from the model's own latent stream. Three signals (increment energy, direction change, prediction novelty) are segmented offline with PELT, and the segmentation is distilled into a causal CNN of under 4k parameters. The detector occupies two posts: at the imagined post it reads a probe rollout and sets the planning horizon, and at the real post it reads the observation stream, where a fire truncates execution and forces a re-anchoring.

## Pipeline (`aiim2026/`)

- `src/e2_extract_features.py` — three boundary signals plus multi-scale rolling statistics
- `src/e3_segment.py` — offline PELT segmentation, with BinSeg / BottomUp / BOCPD controls
- `src/e4_train_head.py` — distillation of the PELT boundaries into a causal 1D CNN
- `src/e5_alignment.py`, `src/e5_diagnosis.py` — offline alignment checks and diagnosis
- `src/e6_eval.py` — end-to-end evaluation on PushT and OGBench Cube
- `src/probe_t_policy.py` — dual-post probe schedule (imagined probe horizon, real-stream re-anchoring)
- `src/run_*.py`, `src/*_arm.py` — fixed-horizon and PACE baselines, grid sweeps
- `results/` — per-seed tables, detector AUC / F1, probe horizon statistics
- `EXPERIMENT_GUIDE.md`, `DESIGN_NOTES.md`, `PROGRESS.md` — pipeline documentation

## Setup

```bash
conda create -n aiim python=3.10 -y
conda activate aiim
pip install -r aiim2026/env_requirements.txt
```

## Main result (PushT)

| setting | d = 50 | d = 75 |
| --- | --- | --- |
| best fixed horizon | 37.1% | 20.9% |
| PACE | 32.4% | 21.1% |
| probe (ours) | **46.2%** | **26.0%** |
