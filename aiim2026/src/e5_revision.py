"""E5 table revision per REVISION_20261103.md §1.

Adds to the E5 story (new file e5_alignment_v2.json — the original
e5_alignment.json is never modified):
  - constant-rhythm baselines: fires from step 3+phi (phi=0..4), every 5
    steps, no features at all; evaluated with the exact E5 matching code
  - AUC rows: the E5 student head's per-step probability scored against
    physical event labels (exact-step and +-2-dilated), on the same 100
    hold episodes
  - narrative rewrite per §1: rhythm-trap disclosure
"""

import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_train_head import TinyGRU, load_dataset
from e5_alignment import f1_event
from fit_features import kinematic_events

ROOT = Path(__file__).resolve().parents[1]
FEAT, MODELS, RESULTS = (ROOT / 'features', ROOT / 'models',
                          ROOT / 'results')
H5 = Path.home() / '.stable_worldmodel/datasets/pusht_expert_train.h5'
FS, MATCH_TOL = 5, 2


def main():
    meta4 = json.load(open(MODELS / 'e4_meta.json'))
    hold_ids = meta4['hold_episode_ids']
    e5 = json.load(open(RESULTS / 'e5_alignment.json'))

    # reference events: exact E5 protocol
    ref = {}
    with h5py.File(H5, 'r', swmr=True) as f:
        el, eo = f['ep_len'][:], f['ep_offset'][:]
        for ep in hold_ids:
            s = f['state'][int(eo[ep]):int(eo[ep]) + int(el[ep])]
            push, settle = kinematic_events(s)
            ref[ep] = np.unique(
                -(-np.nonzero(push | settle)[0] // FS))
    K_of = {ep: int(np.load(ROOT / 'features' / f'ep{ep}.npz')
                   ['model_len']) for ep in hold_ids}

    # ---- constant-rhythm baselines, exact E5 matching ---------------
    rhythm = []
    for phi in range(5):
        f1s = []
        tp = fp = fn = 0
        for ep in hold_ids:
            pred = np.arange(3 + phi, K_of[ep], 5)
            f1, t, p_, n_ = f1_event(pred, ref[ep])
            f1s.append(f1)
            tp += t
            fp += p_
            fn += n_
        rhythm.append({'phase': phi,
                       'macro_f1': float(np.mean(f1s)),
                       'micro_f1': 2 * tp / max(2 * tp + fp + fn, 1),
                       'tp': tp, 'fp': fp, 'fn': fn})
        print(f'rhythm phi={phi}: macro={rhythm[-1]["macro_f1"]:.3f} '
              f'micro={rhythm[-1]["micro_f1"]:.3f}')

    # ---- AUC of the E5 student head ---------------------------------
    model = TinyGRU()
    model.load_state_dict(torch.load(MODELS / 'head_gru.pt',
                                     map_location='cpu'))
    model.eval()
    ps, y_exact, y_dil = [], [], []
    for sig, lab in load_dataset(hold_ids):
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(sig[None])))[0].numpy()
        ps.append(p)
        ep = hold_ids[len(ps) - 1]
        steps = np.arange(3, 3 + len(p))
        ev = ref[ep]
        y_exact.append(np.isin(steps, ev).astype(int))
        dil = np.zeros_like(y_exact[-1])
        for e in ev:
            dil[np.abs(steps - int(e)) <= MATCH_TOL] = 1
        y_dil.append(dil)
    p_all = np.concatenate(ps)
    auc_exact = float(roc_auc_score(np.concatenate(y_exact), p_all))
    auc_dil = float(roc_auc_score(np.concatenate(y_dil), p_all))
    print(f'student AUC vs exact event steps: {auc_exact:.3f}')
    print(f'student AUC vs +-2 dilated:       {auc_dil:.3f}')

    out = {
        'date': '2026-11-03',
        'note': 'revision per REVISION_20261103.md §1; original '
                'e5_alignment.json untouched (rhythm-trap finding is '
                'preserved)',
        'original_rows': e5['methods'],
        'constant_rhythm': rhythm,
        'rhythm_macro_median': float(np.median(
            [r['macro_f1'] for r in rhythm])),
        'rhythm_macro_max': float(max(r['macro_f1'] for r in rhythm)),
        'student_auc_vs_exact_events': auc_exact,
        'student_auc_vs_dilated_events': auc_dil,
        'narrative': (
            'Physical events in this PushT expert data carry an intrinsic '
            'rhythm (~one event per 3.6 model steps, near-uniform): a '
            'constant 5-step timer reaches micro F1 '
            f'{rhythm[0]["micro_f1"]:.3f}-'
            f'{rhythm[-1]["micro_f1"]:.3f} across phases, and neither '
            'PELT (0.650) nor the distilled student (0.645) beats it; '
            'the student per-step probability has near-zero discrimination '
            f'(AUC {auc_exact:.2f} vs exact event steps). F1-style '
            'alignment metrics on this data must report the constant-rhythm '
            'baseline, otherwise they are inflated by rhythm alone.'),
    }
    with open(RESULTS / 'e5_alignment_v2.json', 'w') as f:
        json.dump(out, f, indent=2)
    print('-> results/e5_alignment_v2.json')


if __name__ == '__main__':
    main()
