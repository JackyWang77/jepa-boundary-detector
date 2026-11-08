"""Cube twin of r2_dualhead.py — same 18ch dual-head, cube labels & latents.

Labels: ogb_cube_events(privileged_block_0_pos) -> manip onset + settle,
mapped frame//FS -> model step (same convention as PushT kinematic_events).
Outputs: models/cube_head_r2_dual_{cnn,gru}.pt, models/cube_fz_dual_linear.npy,
         models/cube_r2_meta_dual.json, results/cube_r2_dual.json
"""

import json
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_extract_features import causal_stats
from e4_train_head import CausalCNN, TinyGRU, apply_refractory
from fit_features import ogb_cube_events
from r2_dualhead import (dual_innov, fit_forward, collate, hold_auc,
                         fire_f1, rhythm_baselines, train, WINDOWS, TAU,
                         REFRACT, AUC_GATE, F1_GATE)

ROOT = Path(__file__).resolve().parents[1]
FEAT, MODELS, RESULTS = (ROOT / 'features_cube', ROOT / 'models',
                         ROOT / 'results')
H5 = Path(r'D:\stable_worldmodel\datasets\cube_single_expert.h5')
FS = 5
CTX = 3
SEED = 0
HOLD_FRAC = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_episode(ep, f5, W):
    fd = np.load(FEAT / f'ep{ep}.npz')
    K = int(fd['model_len'])
    T = int(fd['raw_len'])
    off = int(f5['ep_offset'][ep])
    pos = f5['privileged_block_0_pos'][off:off + T]
    onset, settle = ogb_cube_events(pos)
    ev = np.unique(-(-np.nonzero(onset | settle)[0] // FS))
    ev = ev[ev < K]

    f3d = dual_innov(fd['z'], W)
    chans = []
    for name in ('f1', 'f2'):
        raw = fd[name]
        for w in WINDOWS:
            m, sd = causal_stats(raw, w)
            chans.extend([m, sd])
    for w in WINDOWS:
        m, sd = causal_stats(f3d, w)
        chans.extend([m, sd])
    X = np.stack(chans, axis=1)[3:]
    y = np.isin(np.arange(3, K), ev).astype(np.float32)
    return X.astype(np.float32), ev, K, y


def main():
    torch.manual_seed(SEED)
    meta = json.loads((FEAT / 'meta.json').read_text())
    ids = np.array(meta['episode_ids'])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ids))
    n_hold = int(len(ids) * HOLD_FRAC)
    hold_ids, train_ids = ids[perm[:n_hold]], ids[perm[n_hold:]]
    print(f'train {len(train_ids)} eps | hold {len(hold_ids)} eps '
          f'| device {DEVICE}')

    z_train = [np.load(FEAT / f'ep{ep}.npz')['z'] for ep in train_ids]
    W = fit_forward(z_train)
    np.save(MODELS / 'cube_fz_dual_linear.npy', W)
    print(f'F_z linear fit: {W.shape}, {W.size} params')

    with h5py.File(H5, 'r', swmr=True) as f5:
        cache = {ep: load_episode(ep, f5, W) for ep in ids}
    train_eps = [cache[ep] for ep in train_ids]
    hold_eps = [cache[ep] for ep in hold_ids]

    X_all = np.concatenate([x for x, _, _, _ in train_eps])
    mu, sd = X_all.mean(0), X_all.std(0) + 1e-8
    for eps in (train_eps, hold_eps):
        for i, (x, ev, K, y) in enumerate(eps):
            eps[i] = ((x - mu) / sd, ev, K, y)

    n_pos = sum(int(y.sum()) for _, _, _, y in train_eps)
    n_rows = sum(len(y) for _, _, _, y in train_eps)
    data_pw = (n_rows - n_pos) / max(n_pos, 1)
    print(f'train rows={n_rows} positives={n_pos} '
          f'({n_pos / n_rows:.1%}) pos_weight={data_pw:.2f}')

    rhythm = rhythm_baselines(hold_eps)
    rhythm_max = max(r['micro_f1'] for r in rhythm)
    results = {'rhythm_max_micro_f1': rhythm_max,
               'label_rate_train': n_pos / n_rows,
               'data_pos_weight': data_pw, 'sweep': []}

    best = None
    for pw in [1.0, 2.0, round(data_pw, 2), round(min(data_pw, 10.0), 2)]:
        t0 = time.time()
        model = TinyGRU(in_ch=18).to(DEVICE)
        auc, epochs = train(model, train_eps, hold_eps, DEVICE, pw)
        micro, macro, counts = fire_f1(model, hold_eps, DEVICE)
        n_par = sum(p.numel() for p in model.parameters())
        results['sweep'].append(
            {'head': 'gru', 'pos_weight': pw, 'params': n_par,
             'hold_auc': auc, 'fire_micro_f1': micro, 'tp_fp_fn': counts,
             'epochs': epochs, 'seconds': round(time.time() - t0, 1)})
        print(f'[gru pw={pw}] AUC={auc:.3f} F1={micro:.3f}')
        if best is None or auc > best[1]:
            best = (model, auc, {'head': 'gru', 'pos_weight': pw})
    for pw in sorted({round(data_pw, 2), round(min(data_pw, 10.0), 2)}):
        t0 = time.time()
        model = CausalCNN(in_ch=18).to(DEVICE)
        auc, epochs = train(model, train_eps, hold_eps, DEVICE, pw)
        micro, macro, counts = fire_f1(model, hold_eps, DEVICE)
        n_par = sum(p.numel() for p in model.parameters())
        results['sweep'].append(
            {'head': 'cnn', 'pos_weight': pw, 'params': n_par,
             'hold_auc': auc, 'fire_micro_f1': micro, 'tp_fp_fn': counts,
             'epochs': epochs, 'seconds': round(time.time() - t0, 1)})
        print(f'[cnn pw={pw}] AUC={auc:.3f} F1={micro:.3f}')
        if auc > best[1]:
            best = (model, auc, {'head': 'cnn', 'pos_weight': pw})

    sel = max(results['sweep'], key=lambda r: r['hold_auc'])
    results['selected'] = sel
    results['gates'] = {'auc_gate': AUC_GATE, 'f1_gate': F1_GATE,
                        'rhythm_max': rhythm_max}

    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    head_file = MODELS / f'cube_head_r2_dual_{sel["head"]}.pt'
    torch.save(best[0].state_dict(), head_file)
    channels = [f'{n}_{stat}_W{w}' for n in ('f1', 'f2', 'f3_dual')
                for w in WINDOWS for stat in ('mean', 'std')]
    deploy_meta = {
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'task': 'cube',
        'head_class': sel['head'],
        'head_ckpt': str(head_file),
        'params': sel['params'],
        'fz_ckpt': str(MODELS / 'cube_fz_dual_linear.npy'),
        'channels': channels,
        'feat_mu': mu.tolist(), 'feat_sd': sd.tolist(),
        'tau': TAU, 'refractory': REFRACT,
        'label': 'ogb_cube_events(manip_onset|settle) @ frame//5',
        'train_episode_ids': [int(i) for i in train_ids],
        'hold_episode_ids': [int(i) for i in hold_ids],
        'selection': 'max hold AUC over sweep',
        'selected_metrics': sel,
        'gates': results['gates'],
        'action_mean': meta['action_mean'],
        'action_std_plus_eps': meta['action_std_plus_eps'],
    }
    (MODELS / 'cube_r2_meta_dual.json').write_text(
        json.dumps(deploy_meta, indent=2))

    if sel['hold_auc'] >= AUC_GATE and sel['fire_micro_f1'] > F1_GATE:
        results['verdict'] = 'PASS'
    elif sel['hold_auc'] >= 0.65:
        results['verdict'] = 'PARTIAL'
    else:
        results['verdict'] = 'FAIL: AUC<0.65'
    (RESULTS / 'cube_r2_dual.json').write_text(json.dumps(results, indent=2))
    print('selected:', json.dumps(sel))
    print('verdict:', results['verdict'])


if __name__ == '__main__':
    main()
