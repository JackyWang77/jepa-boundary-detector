"""E6 evaluation entry: boundary / equal arms on the PushT protocol.

Mirrors third_party stable-worldmodel scripts/plan/eval_wm.py so the episode
sampling matches the E1 flat baseline exactly (same seed -> same 50 start/goal
tasks per (d, seed)). Differences:
  - policy is wrapped in BoundaryPolicy (arm = boundary | equal)
  - no video saving; per-episode replan logs are written to JSON
Usage (from scripts/plan dir, like run_baseline.sh):
  python e6_eval.py --config-name=pusht.yaml policy=pusht/weights.pt \
    eval.goal_offset_steps=75 eval.eval_budget=150 \
    +plan_config.history_len=3 solver.device=mps seed=42 \
    +arm=boundary +head_path=... +out_json=...
"""

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from boundary_policy import BoundaryPolicy

SWM_PLAN = SRC.parent / 'third_party' / 'stable-worldmodel' / 'scripts' / 'plan'
MODELS_DIR = SRC.parent / 'models'


def localize_ckpt_paths(meta):
    """r2_meta*.json record absolute paths from the training machine; on
    any other box fall back to the same basename under models/."""
    for key in ('head_ckpt', 'fz_ckpt'):
        if key in meta and not Path(meta[key]).exists():
            meta[key] = str(MODELS_DIR / Path(meta[key]).name)
    return meta


def img_transform(cfg, dtype=torch.float32):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def episode_col(dataset):
    names = set(dataset.column_names)
    names |= set(getattr(dataset, '_schema_names', ()))
    return 'episode_idx' if 'episode_idx' in names else 'ep_idx'


@hydra.main(version_base=None,
            config_path=str(SWM_PLAN / 'config'), config_name='pusht')
def run(cfg: DictConfig):
    # Only the executed part of a plan must fit the budget. With the
    # decoupled arm (horizon=15, receding=5) the plan is longer than the
    # receding exec length on purpose; the tail is warm-start material.
    # NOTE: downgraded to warning — flat(15,15)·d25 has receding*action_block=75
    # vs budget=50, meaning "plan once, execute to end with no replan". Valid data.
    _recede_exec = cfg.plan_config.receding_horizon * cfg.plan_config.action_block
    if _recede_exec > cfg.eval.eval_budget:
        print(f"[WARN] receding_exec={_recede_exec} > budget={cfg.eval.eval_budget} "
              f"→ no replan will happen within budget window (plan-once arm)")

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {'pixels': img_transform(cfg, img_dtype),
                  'goal': img_transform(cfg, img_dtype)}

    dataset = swm.data.load_dataset(
        cfg.eval.dataset_name, cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    col_name = episode_col(dataset)
    ep_indices, _ = np.unique(dataset.get_col_data(col_name),
                              return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != 'action':
            process[f'goal_{col}'] = process[col]

    model = swm.wm.utils.load_pretrained(cfg.policy)
    model = model.to(cfg.solver.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    config = swm.PlanConfig(**cfg.plan_config)
    # one-time record: the plan/exec split defines the arm (paper setup
    # section cites this line + the CEM solve-time ratio as h-15 evidence)
    print(f'[plan] horizon={config.horizon} '
          f'receding={config.receding_horizon} '
          f'action_block={config.action_block} arm='
          f'{cfg.get("arm", "flat")}: plan {config.horizon} blocks '
          f'({config.horizon * config.action_block} env steps), '
          f'exec/re-anchor every {config.receding_horizon} blocks')
    objective = hydra.utils.instantiate(cfg.objective)
    cost = swm.planning.ShootingCostEvaluator(model, objective)
    solver = hydra.utils.instantiate(cfg.solver, cost=cost)

    arm = cfg.get('arm', 'flat')
    head_path = cfg.get('head_path', None)
    if arm in ('boundary', 'equal', 'decoupled'):
        feats_meta = json.load(
            open(SRC.parent / 'features' / 'meta.json'))
        head_cfg = {}
        if arm == 'boundary':
            r2m = localize_ckpt_paths(
                json.load(open(SRC.parent / 'models' / 'r2_meta.json')))
            head_cfg = dict(
                head_ckpt=r2m['head_ckpt'],
                head_class=r2m['head_class'],
                channels=r2m['channels'],
                feat_mu=r2m['feat_mu'], feat_sd=r2m['feat_sd'],
                tau=r2m['tau'], refractory=r2m['refractory'],
            )
        policy = BoundaryPolicy(
            solver=solver, config=config, process=process,
            transform=transform, model=model,
            action_mean=feats_meta['action_mean'],
            action_std=feats_meta['action_std_plus_eps'],
            arm=arm, interval=cfg.get('interval', 4),
            **head_cfg,
        )
    elif arm == 'probe_t':
        from probe_t_policy import ProbeTPolicy
        feats_dir = cfg.get('features_dir', 'features')
        det_meta = cfg.get('detector_meta', 'r2_meta_dual.json')
        feats_meta = json.load(
            open(SRC.parent / feats_dir / 'meta.json'))
        r2m = localize_ckpt_paths(
            json.load(open(SRC.parent / 'models' / det_meta)))
        policy = ProbeTPolicy(
            solver=solver, config=config, process=process,
            transform=transform, model=model,
            head_ckpt=r2m['head_ckpt'],
            channels=r2m['channels'],
            feat_mu=r2m['feat_mu'], feat_sd=r2m['feat_sd'],
            fz_path=r2m.get('fz_ckpt',
                            str(SRC.parent / 'models' / 'fz_dual_linear.npy')),
            tau=r2m['tau'], refractory=r2m['refractory'],
            action_mean=feats_meta['action_mean'],
            action_std=feats_meta['action_std_plus_eps'],
            head_class=r2m.get('head_class', 'cnn'),
            t_plus_one=bool(cfg.get('t_plus_one', False)),
        )
    else:
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process,
            transform=transform,
        )

    # --- episode selection: verbatim from eval_wm.py (same seed = same tasks)
    episode_len = [np.max(dataset.get_col_data('step_idx')[
        dataset.get_col_data(col_name) == ep]) + 1 for ep in ep_indices]
    episode_len = np.array(episode_len)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i]
                          for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id]
         for ep_id in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices), size=cfg.eval.num_eval, replace=False)
    random_episode_indices = np.sort(valid_indices[
        random_episode_indices])
    eval_episodes = dataset.get_col_data(col_name)[random_episode_indices]
    eval_start_idx = dataset.get_col_data('step_idx')[random_episode_indices]

    world.set_policy(policy)

    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(
            cfg.eval.get('callables'), resolve=True),
        video=cfg.get('video_dir', None),
    )
    print(metrics)

    logs = policy.finalize() if hasattr(policy, 'finalize') else {}
    out = {
        'arm': arm,
        'goal_offset_steps': int(cfg.eval.goal_offset_steps),
        'eval_budget': int(cfg.eval.eval_budget),
        'seed': int(cfg.seed),
        'interval': int(cfg.get('interval', 0)),
        'success_rate': float(metrics['success_rate']),
        'episode_successes': [bool(x) for x in
                               np.asarray(metrics['episode_successes'])],
        'replan_logs': {str(i): v for i, v in enumerate(logs)},
    }
    out_path = Path(cfg.get('out_json', 'e6_out.json'))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    n_replans = sum(len(ep['replans'])
                    for eps in logs for ep in eps)
    n_fires = sum(len(ep['fires']) for eps in logs for ep in eps)
    print(f'[{arm}] replans={n_replans} fires={n_fires} '
          f'-> {out_path}')


if __name__ == '__main__':
    run()
