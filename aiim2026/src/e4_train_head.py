"""E4: distill the PELT teacher into a lightweight online detection head.

Per EXPERIMENT_GUIDE.md: two candidate heads, both trained, report the
better one.
  - CausalCNN: 2 layers x 16 channels, kernel 5, causal left-pad
               (~1.8k params, receptive field ~9 steps)
  - TinyGRU:   unidirectional GRU, hidden 16 (~1.2k params)
Training: BCE + pos_weight (computed from the actual label imbalance),
early stop on the pseudo-label F1 of 20% held-out trajectories (seeded
split, ids recorded in meta). Inference rule: p > tau=0.5 + 5-step
refractory; F1 matching tolerance +-2 steps.
Output: aiim2026/models/head_{cnn,gru}.pt + e4_meta.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT = Path(__file__).resolve().parents[1] / 'features'
SEG = Path(__file__).resolve().parents[1] / 'segments' / 'e3'
OUT = Path(__file__).resolve().parents[1] / 'models'
CHANNELS = ['f1_mean', 'f1_std', 'f2_mean', 'f2_std', 'f3_mean', 'f3_std']
HEAD = 3
TAU = 0.5
REFRACT = 5
MATCH_TOL = 2
HOLD_FRAC = 0.2
SEED = 0
MAX_EPOCHS, PATIENCE, LR, BATCH = 300, 25, 1e-3, 32


class CausalCNN(nn.Module):
    def __init__(self, in_ch=6, ch=16, kernel=5):
        super().__init__()
        self.pad = kernel - 1
        self.conv1 = nn.Conv1d(in_ch, ch, kernel)
        self.conv2 = nn.Conv1d(ch, ch, kernel)
        self.head = nn.Linear(ch, 1)

    def forward(self, x):                 # x: (B, T, C)
        x = x.transpose(1, 2)              # (B, C, T)
        x = F.pad(x, (self.pad, 0))
        x = F.relu(self.conv1(x))
        x = F.pad(x, (self.pad, 0))
        x = F.relu(self.conv2(x))
        return self.head(x.transpose(1, 2)).squeeze(-1)   # (B, T)


class TinyGRU(nn.Module):
    def __init__(self, in_ch=6, hidden=16):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)


def load_dataset(ep_ids):
    """Per-episode (signal, label) arrays on the trimmed model-step base."""
    data = []
    for ep in ep_ids:
        fd = np.load(FEAT / f'ep{ep}.npz')
        sd = np.load(SEG / f'ep{ep}.npz')
        sig = np.stack([fd[c] for c in CHANNELS], axis=1)[HEAD:]
        lab = sd['labels_pelt'][HEAD:].astype(np.float32)
        assert len(sig) == len(lab)
        data.append((sig.astype(np.float32), lab))
    return data


def collate(batch):
    L = max(len(s) for s, _ in batch)
    B = len(batch)
    x = np.zeros((B, L, len(CHANNELS)), np.float32)
    y = np.full((B, L), -100.0, np.float32)     # ignore-index in mask
    m = np.zeros((B, L), bool)
    for i, (s, l) in enumerate(batch):
        x[i, :len(s)] = s
        y[i, :len(l)] = l
        m[i, :len(l)] = True
    return (torch.from_numpy(x), torch.from_numpy(y),
            torch.from_numpy(m))


def predict_steps(model, sig, device):
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(sig[None]).to(device)
        return torch.sigmoid(model(x))[0].cpu().numpy()


def apply_refractory(pred_bool):
    out = pred_bool.copy()
    last = -REFRACT - 1
    for t in np.nonzero(pred_bool)[0]:
        if t - last >= REFRACT:
            out[t] = True
            last = t
        else:
            out[t] = False
    return out


def episode_f1(pred_p, labels, tau=TAU):
    """Event-level F1: predicted boundaries (tau + refractory) vs PELT
    pseudo-labels, +-tol matching, greedy one-to-one."""
    pred = pred_p > tau
    pred = apply_refractory(pred)
    lab_idx = np.nonzero(labels)[0]
    pred_idx = np.nonzero(pred)[0]
    if len(lab_idx) == 0 and len(pred_idx) == 0:
        return 1.0
    matched = set()
    tp = 0
    for p in pred_idx:
        near = [l for l in lab_idx if abs(int(p) - int(l)) <= MATCH_TOL
                and l not in matched]
        if near:
            tp += 1
            matched.add(near[0])
    fp = len(pred_idx) - tp
    fn = len(lab_idx) - tp
    return 2 * tp / max(2 * tp + fp + fn, 1)


def evaluate(model, data, device):
    f1s = []
    for sig, lab in data:
        p = predict_steps(model, sig, device)
        f1s.append(episode_f1(p, lab))
    return float(np.mean(f1s))


def train(model, train_data, hold_data, device, pos_weight):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device))
    best_f1, best_state, curve, bad = -1, None, [], 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        losses = []
        order = np.random.default_rng(epoch).permutation(len(train_data))
        for lo in range(0, len(order), BATCH):
            batch = [train_data[i] for i in order[lo:lo + BATCH]]
            x, y, m = collate(batch)
            x, y, m = x.to(device), y.to(device), m.to(device)
            logits = model(x)
            mask = m.float()
            loss = (lossf(logits * mask, torch.clamp(y, 0, 1)) * mask
                    ).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        f1 = evaluate(model, hold_data, device)
        curve.append({'epoch': epoch,
                      'train_loss': float(np.mean(losses)),
                      'hold_f1': f1})
        if f1 > best_f1:
            best_f1, best_state, bad = f1, \
                {k: v.detach().clone() for k, v in
                 model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return best_f1, curve


def main(smoke=False):
    OUT.mkdir(parents=True, exist_ok=True)
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.manual_seed(SEED)

    meta_feats = json.load(open(FEAT / 'meta.json'))
    ids = meta_feats['episode_ids']
    if smoke:
        ids = ids[:24]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ids))
    n_hold = max(1, int(len(ids) * HOLD_FRAC))
    hold_ids = [ids[i] for i in perm[:n_hold]]
    train_ids = [ids[i] for i in perm[n_hold:]]

    train_data = load_dataset(train_ids)
    hold_data = load_dataset(hold_ids)

    n_pos = sum(int(l.sum()) for _, l in train_data)
    n_neg = sum(len(l) for _, l in train_data) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f'train={len(train_ids)} eps ({n_pos} pos / {n_neg} neg, '
          f'pos_weight={pos_weight:.1f}), hold={len(hold_ids)} eps')

    results = {}
    for name, mk in [('cnn', lambda: CausalCNN()),
                     ('gru', lambda: TinyGRU())]:
        model = mk().to(dev)
        n_params = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        f1, curve = train(model, train_data, hold_data, dev, pos_weight)
        torch.save(model.state_dict(), OUT / f'head_{name}.pt')
        tr_f1 = evaluate(model, train_data, dev)
        results[name] = {'params': n_params, 'hold_f1': f1,
                         'train_f1': tr_f1, 'curve': curve,
                         'train_seconds': round(time.time() - t0, 1)}
        print(f'[{name}] params={n_params} hold F1={f1:.3f} '
              f'train F1={tr_f1:.3f} epochs={len(curve)}')

    selected = max(results, key=lambda k: results[k]['hold_f1'])
    meta = {
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'experiment': 'E4 student head distillation',
        'train_episode_ids': [int(i) for i in train_ids],
        'hold_episode_ids': [int(i) for i in hold_ids],
        'pos_weight': float(pos_weight),
        'label_imbalance_note': 'guide estimated ~1:30; actual PELT label '
            f'ratio is 1:{pos_weight:.1f}, pos_weight set from data',
        'tau': TAU, 'refractory_steps': REFRACT,
        'match_tolerance': MATCH_TOL, 'split_seed': SEED,
        'selected_head': selected,
        'results': {k: {kk: vv for kk, vv in v.items()}
                    for k, v in results.items()},
        'smoke': smoke,
    }
    suffix = '_smoke' if smoke else ''
    with open(OUT / f'e4_meta{suffix}.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'selected head: {selected} -> models/head_{selected}.pt')


if __name__ == '__main__':
    import sys
    main(smoke='--smoke' in sys.argv)
