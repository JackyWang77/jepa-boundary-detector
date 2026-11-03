"""R2 detector rescue per REVISION_20261103.md §3 (privileged distillation).

- labels: exact model-step event labels (push_onset | settle from the raw
  simulator state, mapped via ceil(t/5), dedup), NO dilation
- features: 18 channels = {f1, f2, f3} x {mean, std} x W in {2, 4, 8},
  computed from the stored raw streams in features/ep{id}.npz, rows from
  model step 3; z-scored with train-split statistics
- training: TinyGRU(18) + CausalCNN(18), pos_weight sweep, early stop on
  hold AUC (exact event labels) — never pseudo-label F1
- gates (pre-registered): hold AUC >= 0.70 AND refractory fire micro F1
  > 0.745 (= rhythm baseline 0.715 + 0.03)
- output: results/r2_rescue.json (new file; all original results untouched)
"""

import json
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_extract_features import causal_stats
from e4_train_head import CausalCNN, TinyGRU, apply_refractory
from fit_features import kinematic_events

ROOT = Path(__file__).resolve().parents[1]
FEAT, MODELS, RESULTS = (ROOT / 'features', ROOT / 'models',
                          ROOT / 'results')
H5 = Path.home() / '.stable_worldmodel/datasets/pusht_expert_train.h5'
FS = 5
WINDOWS = (2, 4, 8)
TAU, REFRACT, MATCH_TOL = 0.5, 5, 2
MAX_EPOCHS, PATIENCE, LR, BATCH, SEED = 300, 25, 1e-3, 32, 0
F1_GATE = 0.745
AUC_GATE = 0.70


def load_episode(ep, f5):
    """Exact event labels + 18-channel rows for one episode."""
    fd = np.load(FEAT / f'ep{ep}.npz')
    K = int(fd['model_len'])
    T = int(fd['raw_len'])
    off = int(f5['ep_offset'][ep])
    s = f5['state'][off:off + T]
    push, settle = kinematic_events(s)
    ev = np.unique(-(-np.nonzero(push | settle)[0] // FS))
    ev = ev[ev < K]

    chans = []
    for name in ('f1', 'f2', 'f3'):
        raw = fd[name]
        for w in WINDOWS:
            m, sd = causal_stats(raw, w)
            chans.extend([m, sd])
    X = np.stack(chans, axis=1)[3:]                  # (K-3, 18)
    y = np.isin(np.arange(3, K), ev).astype(np.float32)
    return X.astype(np.float32), ev, K, y


def collate(batch):
    L = max(len(x) for x, _, _, _ in batch)
    B = len(batch)
    X = np.zeros((B, L, batch[0][0].shape[1]), np.float32)
    M = np.zeros((B, L), bool)
    Y = np.zeros((B, L), np.float32)
    for i, (x, _, _, y) in enumerate(batch):
        X[i, :len(x)] = x
        M[i, :len(x)] = True
        Y[i, :len(y)] = y
    return (torch.from_numpy(X), torch.from_numpy(M),
            torch.from_numpy(Y))


def run_head(model, episodes, device):
    """Per-episode probability rows."""
    ps = []
    model.eval()
    with torch.no_grad():
        for X, ev, K, _ in episodes:
            x = torch.from_numpy(X)[None].to(device)
            ps.append(torch.sigmoid(model(x))[0].cpu().numpy())
    return ps


def hold_auc(model, episodes, device):
    ys = np.concatenate([y.astype(int) for _, _, _, y in episodes])
    p = np.concatenate(run_head(model, episodes, device))
    return float(roc_auc_score(ys, p)) if 0 < ys.sum() < len(ys) else 0.5


def fire_f1(model, episodes, device):
    """tau + refractory fires vs exact events, +-2 greedy matching."""
    tp = fp = fn = 0
    macro = []
    for p, (X, ev, K, _) in zip(run_head(model, episodes, device),
                                episodes):
        pred = np.nonzero(apply_refractory(p > TAU))[0] + 3
        matched = set()
        t = 0
        for q in pred:
            near = [e for e in ev
                    if abs(int(q) - int(e)) <= MATCH_TOL and e not in matched]
            if near:
                matched.add(near[0])
                t += 1
        tp += t
        fp += len(pred) - t
        fn += len(ev) - t
        denom = len(pred) + len(ev)
        macro.append(2 * t / denom if denom else 1.0)
    micro = 2 * tp / max(2 * tp + fp + fn, 1)
    return float(micro), float(np.mean(macro)), (tp, fp, fn)


def rhythm_baselines(episodes):
    """Constant rhythm from step 3+phi every 5 steps, same matching."""
    out = []
    for phi in range(5):
        tp = fp = fn = 0
        for _, ev, K, _ in episodes:
            pred = np.arange(3 + phi, K, 5)
            matched = set()
            t = 0
            for q in pred:
                near = [e for e in ev
                        if abs(int(q) - int(e)) <= MATCH_TOL
                        and e not in matched]
                if near:
                    matched.add(near[0])
                    t += 1
            tp += t
            fp += len(pred) - t
            fn += len(ev) - t
        out.append({'phase': phi,
                    'micro_f1': 2 * tp / max(2 * tp + fp + fn, 1),
                    'tp': tp, 'fp': fp, 'fn': fn})
    return out


def train(model, train_eps, hold_eps, device, pos_weight):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))
    best_auc, best_state, bad = -1, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = np.random.default_rng(SEED + epoch).permutation(
            len(train_eps))
        for lo in range(0, len(order), BATCH):
            batch = [train_eps[i] for i in order[lo:lo + BATCH]]
            X, M, Y = collate(batch)
            X, M, Y = X.to(device), M.to(device), Y.to(device)
            logits = model(X)
            y = Y
            mask = M.float()
            loss = (lossf(logits * mask, y * mask) * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        auc = hold_auc(model, hold_eps, device)
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return best_auc, epoch + 1


def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.manual_seed(SEED)

    meta4 = json.load(open(MODELS / 'e4_meta.json'))
    train_ids, hold_ids = (meta4['train_episode_ids'],
                           meta4['hold_episode_ids'])

    with h5py.File(H5, 'r', swmr=True) as f5:
        cache = {ep: load_episode(ep, f5)
                 for ep in train_ids + hold_ids}
    train_eps = [cache[ep] for ep in train_ids]
    hold_eps = [cache[ep] for ep in hold_ids]

    # z-score channels with train statistics
    X_all = np.concatenate([x for x, _, _, _ in train_eps])
    mu, sd = X_all.mean(0), X_all.std(0) + 1e-8
    for eps in (train_eps, hold_eps):
        for i, (x, ev, K, y) in enumerate(eps):
            eps[i] = ((x - mu) / sd, ev, K, y)

    n_pos = sum(int(y.sum()) for _, _, _, y in train_eps)
    n_rows = sum(len(y) for _, _, _, y in train_eps)
    data_pw = (n_rows - n_pos) / max(n_pos, 1)
    print(f'train rows={n_rows} positives={n_pos} '
          f'({n_pos / n_rows:.1%})  data pos_weight={data_pw:.2f}')

    rhythm = rhythm_baselines(hold_eps)
    rhythm_max = max(r['micro_f1'] for r in rhythm)
    print('rhythm baselines (hold, micro F1):',
          [round(r['micro_f1'], 3) for r in rhythm])

    results = {'rhythm_baselines_hold': rhythm,
               'rhythm_max_micro_f1': rhythm_max,
               'label_rate_train': n_pos / n_rows,
               'data_pos_weight': data_pw,
               'sweep': []}

    best = None
    for pw in [1.0, 1.5, 2.0, round(data_pw, 2), 3.0]:
        t0 = time.time()
        model = TinyGRU(in_ch=18).to(device)
        auc, epochs = train(model, train_eps, hold_eps, device, pw)
        micro, macro, counts = fire_f1(model, hold_eps, device)
        n_par = sum(p.numel() for p in model.parameters())
        results['sweep'].append(
            {'head': 'gru', 'pos_weight': pw, 'params': n_par,
             'hold_auc': auc, 'fire_micro_f1': micro,
             'fire_macro_f1': macro, 'tp_fp_fn': counts,
             'epochs': epochs, 'seconds': round(time.time() - t0, 1)})
        print(f'[gru pw={pw}] hold AUC={auc:.3f} fire F1={micro:.3f} '
              f'({epochs} ep, {time.time() - t0:.0f}s)')
        if best is None or auc > best[1]:
            best = (model, auc, {'head': 'gru', 'pos_weight': pw})
    # CNN at the data pos_weight and at the best GRU pos_weight
    for pw in sorted({round(data_pw, 2), best[2]['pos_weight']}):
        t0 = time.time()
        model = CausalCNN(in_ch=18).to(device)
        auc, epochs = train(model, train_eps, hold_eps, device, pw)
        micro, macro, counts = fire_f1(model, hold_eps, device)
        n_par = sum(p.numel() for p in model.parameters())
        results['sweep'].append(
            {'head': 'cnn', 'pos_weight': pw, 'params': n_par,
             'hold_auc': auc, 'fire_micro_f1': micro,
             'fire_macro_f1': macro, 'tp_fp_fn': counts,
             'epochs': epochs, 'seconds': round(time.time() - t0, 1)})
        print(f'[cnn pw={pw}] hold AUC={auc:.3f} fire F1={micro:.3f}')
        if auc > best[1]:
            best = (model, auc, {'head': 'cnn', 'pos_weight': pw})

    sel = max(results['sweep'], key=lambda r: r['hold_auc'])
    results['selected'] = sel
    results['gates'] = {'auc_gate': AUC_GATE, 'f1_gate': F1_GATE,
                        'rhythm_max': rhythm_max}

    # ---- P1-7 fix: persist the selected head + deployment meta --------
    head_file = MODELS / f'head_r2_{sel["head"]}.pt'
    torch.save(best[0].state_dict(), head_file)
    channels = [f'{n}_{stat}_W{w}' for n in ('f1', 'f2', 'f3')
                for w in WINDOWS for stat in ('mean', 'std')]
    deploy_meta = {
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'head_class': sel['head'],
        'head_ckpt': str(head_file),
        'params': sel['params'],
        'pos_weight': sel['pos_weight'],
        'channels': channels,
        'feat_mu': mu.tolist(),
        'feat_sd': sd.tolist(),
        'tau': TAU,
        'refractory': REFRACT,
        'label': 'exact model-step events (push_onset|settle, ceil(t/5))',
        'train_episode_ids': [int(i) for i in train_ids],
        'hold_episode_ids': [int(i) for i in hold_ids],
        'selection': 'max hold AUC over sweep',
        'selected_metrics': sel,
        'gates': results['gates'],
    }
    with open(MODELS / 'r2_meta.json', 'w') as f:
        json.dump(deploy_meta, f, indent=2)
    print(f'saved head -> {head_file} + models/r2_meta.json')
    if sel['hold_auc'] >= AUC_GATE and sel['fire_micro_f1'] > F1_GATE:
        results['verdict'] = ('PASS: boundary arm uses this head '
                              '(AUC>=0.70 and fire F1>0.745)')
    elif sel['hold_auc'] >= 0.65:
        results['verdict'] = ('PARTIAL: 0.65<=AUC<0.70 or F1 gate missed '
                              '-> boundary arm reports partial signal, '
                              'decoupled carries the positive result')
    else:
        results['verdict'] = 'FAIL: AUC<0.65, detector out'
    print('\nselected:', json.dumps(sel, indent=2))
    print('verdict:', results['verdict'])

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / 'r2_rescue.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('-> results/r2_rescue.json')


if __name__ == '__main__':
    main()
