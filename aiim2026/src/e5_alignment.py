"""E5-lite: alignment main table (Table I).

Event-level F1 (+-2 model-step tolerance, greedy one-to-one matching)
against the programmatic reference labels (push onset / settle, mapped
from raw steps to model steps via ceil(t/5)) on the 100 held-out
episodes (never seen by the E4 student):
  - PELT teacher cps
  - student head (GRU, tau=0.5 + 5-step refractory, causal)
  - lower-bound controls: random instants / equal intervals, matched to
    the PELT cp count per episode (seeded)
Output: results/e5_alignment.json + console table + narrative verdict.
"""

import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_train_head import TinyGRU, CHANNELS, HEAD, TAU, REFRACT
from e4_train_head import apply_refractory, load_dataset
from fit_features import kinematic_events

ROOT = Path(__file__).resolve().parents[1]
FEAT, SEG, MODELS = ROOT / 'features', ROOT / 'segments' / 'e3', ROOT / 'models'
RESULTS = ROOT / 'results'
H5 = Path.home() / '.stable_worldmodel/datasets/pusht_expert_train.h5'
FS = 5
MATCH_TOL = 2


def f1_event(pred_idx, lab_idx, tol=MATCH_TOL):
    """Greedy one-to-one matching within +-tol."""
    matched, tp = set(), 0
    for p in pred_idx:
        near = [l for l in lab_idx
                if abs(int(p) - int(l)) <= tol and l not in matched]
        if near:
            tp += 1
            matched.add(near[0])
    fp = len(pred_idx) - tp
    fn = len(lab_idx) - tp
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 1.0, tp, fp, fn


def main():
    RESULTS.mkdir(exist_ok=True)
    meta4 = json.load(open(MODELS / 'e4_meta.json'))
    hold_ids = meta4['hold_episode_ids']
    print(f'hold episodes: {len(hold_ids)} (never seen by the student)')

    # reference events: raw-step events -> model steps (ceil(t/FS))
    ref = {}
    with h5py.File(H5, 'r', swmr=True) as f:
        el, eo = f['ep_len'][:], f['ep_offset'][:]
        for ep in hold_ids:
            s = f['state'][int(eo[ep]):int(eo[ep]) + int(el[ep])]
            push, settle = kinematic_events(s)
            ev_raw = np.nonzero(push | settle)[0]
            ref[ep] = np.unique(-(-ev_raw // FS))      # ceil
    n_events = sum(len(v) for v in ref.values())
    print(f'reference events: {n_events} '
          f'({n_events / len(hold_ids):.1f}/episode)')

    # student head predictions
    model = TinyGRU()
    model.load_state_dict(torch.load(MODELS / 'head_gru.pt',
                                     map_location='cpu'))
    model.eval()
    student_pred = {}
    for sig, lab in load_dataset(hold_ids):
        ep = None
        with torch.no_grad():
            p = torch.sigmoid(model(
                torch.from_numpy(sig[None])))[0].numpy()
        pred = apply_refractory(p > TAU)
        student_pred[hold_ids[len(student_pred)]] = np.nonzero(pred)[0]

    rng = np.random.default_rng(7)
    methods = {'pelt': [], 'student': [], 'random': [], 'equal': []}
    totals = {m: [0, 0, 0] for m in methods}      # tp, fp, fn (micro)
    for ep in hold_ids:
        lab_idx = ref[ep]
        d3 = np.load(SEG / f'ep{ep}.npz')
        cps = d3['cps_pelt']
        K = int(d3['model_len'])
        n = len(cps)
        equal = np.round(np.linspace(HEAD, K - 1, n)).astype(int) \
            if n else np.array([], int)
        rand = rng.choice(np.arange(HEAD, K), size=n,
                          replace=False) if n else np.array([], int)
        for name, pred in [('pelt', cps), ('student', student_pred[ep]),
                           ('random', np.sort(rand)),
                           ('equal', equal)]:
            f1, tp, fp, fn = f1_event(pred, lab_idx)
            methods[name].append(f1)
            totals[name][0] += tp
            totals[name][1] += fp
            totals[name][2] += fn

    print(f'\n{"method":<10} {"macro F1":>9} {"micro F1":>9} '
          f'{"TP":>5} {"FP":>5} {"FN":>5}')
    out = {'n_hold_episodes': len(hold_ids), 'n_events': n_events,
           'tolerance_model_steps': MATCH_TOL, 'methods': {}}
    for name in ('pelt', 'student', 'random', 'equal'):
        tp, fp, fn = totals[name]
        micro = 2 * tp / max(2 * tp + fp + fn, 1)
        macro = float(np.mean(methods[name]))
        print(f'{name:<10} {macro:>9.3f} {micro:>9.3f} '
              f'{tp:>5} {fp:>5} {fn:>5}')
        out['methods'][name] = {'macro_f1': macro, 'micro_f1': micro,
                                'tp': tp, 'fp': fp, 'fn': fn}

    pel = out['methods']['pelt']['macro_f1']
    out['narrative'] = ('high-alignment: change-point segmentation IS '
                        'physical events' if pel > 0.7 else
                        'statistical-dynamics segmentation (degraded '
                        'wording), success-rate test still runs')
    out['narrative_threshold'] = 0.7
    with open(RESULTS / 'e5_alignment.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nnarrative branch: {out["narrative"]}')
    print('-> results/e5_alignment.json')


if __name__ == '__main__':
    main()
