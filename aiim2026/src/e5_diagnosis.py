"""E5 diagnosis (read-only, no parameter changes).

Items:
 1. per-event-type breakdown (push onset vs settle): precision/recall/F1
 2. random / equal control TP/FP/FN (same matching code as the main table),
    plus the count-matching question and an event-count-matched random
 3. PELT cp density vs event density per episode
 4. feature profile of the 65 unmatched cps (FP) vs matched events
 5. temporal position of missed events (FN) within episodes
All computed on the same 100 held-out episodes as results/e5_alignment.json.
"""

import json
from collections import Counter
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_train_head import TinyGRU, load_dataset, apply_refractory
from fit_features import kinematic_events

ROOT = Path(__file__).resolve().parents[1]
FEAT, SEG, MODELS = ROOT / 'features', ROOT / 'segments' / 'e3', ROOT / 'models'
H5 = Path.home() / '.stable_worldmodel/datasets/pusht_expert_train.h5'
FS, TOL = 5, 2


def match(pred_idx, lab_idx, tol=TOL):
    matched, tp = set(), 0
    pairs = []
    for p in pred_idx:
        near = [l for l in lab_idx
                if abs(int(p) - int(l)) <= tol and l not in matched]
        if near:
            matched.add(near[0])
            tp += 1
            pairs.append((int(p), int(near[0])))
    return tp, len(pred_idx) - tp, len(lab_idx) - tp, pairs


def main():
    meta4 = json.load(open(MODELS / 'e4_meta.json'))
    hold_ids = meta4['hold_episode_ids']
    e5 = json.load(open(ROOT / 'results' / 'e5_alignment.json'))

    # reference events by type, features, cps
    ref = {}
    with h5py.File(H5, 'r', swmr=True) as f:
        el, eo = f['ep_len'][:], f['ep_offset'][:]
        for ep in hold_ids:
            s = f['state'][int(eo[ep]):int(eo[ep]) + int(el[ep])]
            push, settle = kinematic_events(s)
            ref[ep] = {
                'push': np.unique(-(-np.nonzero(push)[0] // FS)),
                'settle': np.unique(-(-np.nonzero(settle)[0] // FS)),
            }

    model = TinyGRU()
    model.load_state_dict(torch.load(MODELS / 'head_gru.pt',
                                     map_location='cpu'))
    model.eval()
    student = {}
    for sig, lab in load_dataset(hold_ids):
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(sig[None])))[0].numpy()
        student[hold_ids[len(student)]] = np.nonzero(
            apply_refractory(p > 0.5))[0]

    # ---- 1. per-event-type breakdown ----------------------------------
    print('=' * 70)
    print('1. per-type breakdown (union-matched pairs, typed by the '
          'matched event)')
    stats = {m: {'tp': Counter(), 'fn': Counter(), 'fp': 0}
             for m in ('pelt', 'student')}
    fp_steps, tp_feat, fp_feat = [], [], []
    fn_pos = {'push': [], 'settle': []}
    for ep in hold_ids:
        d3 = np.load(SEG / f'ep{ep}.npz')
        K = int(d3['model_len'])
        union = np.sort(np.concatenate([ref[ep]['push'],
                                        ref[ep]['settle']]))
        fd = np.load(FEAT / f'ep{ep}.npz')
        for m, pred in [('pelt', d3['cps_pelt']),
                        ('student', student[ep])]:
            tp, fp, fn, pairs = match(pred, union)
            stats[m]['fp'] += fp
            got = set()
            for p, l in pairs:
                t = 'push' if l in set(ref[ep]['push'].tolist()) \
                    else 'settle'
                stats[m]['tp'][t] += 1
                got.add(l)
                if m == 'pelt':
                    tp_feat.append((fd['f1'][l], fd['f2'][l],
                                    fd['f3'][l] if l >= 3 else np.nan))
            for l in union:
                if l not in got:
                    t = 'push' if l in set(ref[ep]['push'].tolist()) \
                        else 'settle'
                    stats[m]['fn'][t] += 1
                    if m == 'pelt':
                        fn_pos[t].append(l / max(K - 1, 1))
            if m == 'pelt':
                for p in pred:
                    if p not in [pp for pp, _ in pairs]:
                        fp_steps.append((ep, int(p), l / max(K - 1, 1)
                                         if False else int(p) / max(K - 1, 1)))
                        fp_feat.append((fd['f1'][p], fd['f2'][p],
                                        fd['f3'][p] if p >= 3 else np.nan))
    n_push = sum(len(ref[ep]['push']) for ep in hold_ids)
    n_settle = sum(len(ref[ep]['settle']) for ep in hold_ids)
    print(f'events: push_onset={n_push}, settle={n_settle}, '
          f'total={n_push + n_settle}')
    for m in ('pelt', 'student'):
        n_pred = {'pelt': sum(len(np.load(SEG / f"ep{ep}.npz")['cps_pelt'])
                              for ep in hold_ids),
                  'student': sum(len(student[ep]) for ep in hold_ids)}[m]
        for t, n_ev in [('push', n_push), ('settle', n_settle)]:
            tp = stats[m]['tp'][t]
            fn = stats[m]['fn'][t]
            prec = tp / n_pred
            rec = tp / n_ev
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            print(f'[{m:7}] {t:6}: TP={tp:3}  FN={fn:3}  '
                  f'prec(vs all preds)={prec:.3f}  recall={rec:.3f}  '
                  f'F1={f1:.3f}')
        print(f'[{m:7}] FP total={stats[m]["fp"]}')

    # ---- 2. controls: existing numbers + count-matching + event-matched --
    print('=' * 70)
    print('2. controls')
    for m in ('random', 'equal'):
        s = e5['methods'][m]
        print(f'[{m}] (already in e5_alignment.json): '
              f'TP={s["tp"]} FP={s["fp"]} FN={s["fn"]} '
              f'macro={s["macro_f1"]:.3f} micro={s["micro_f1"]:.3f}')
    n_cps = sum(len(np.load(SEG / f'ep{ep}.npz')['cps_pelt'])
                for ep in hold_ids)
    n_ev = sum(len(ref[ep]['push']) + len(ref[ep]['settle'])
               for ep in hold_ids)
    print(f'count matching: PELT cps total={n_cps} '
          f'({n_cps / len(hold_ids):.2f}/ep), events total={n_ev} '
          f'({n_ev / len(hold_ids):.2f}/ep) -> random/equal controls were '
          f'generated with n = PELT cp count per episode, NOT event count')
    rng = np.random.default_rng(7)
    tp = fp = fn = 0
    for ep in hold_ids:
        d3 = np.load(SEG / f'ep{ep}.npz')
        K = int(d3['model_len'])
        n = len(ref[ep]['push']) + len(ref[ep]['settle'])
        rand = np.sort(rng.choice(np.arange(3, K), size=n,
                                  replace=False)) if n else np.array([], int)
        a, b, c, _ = match(rand, np.sort(
            np.concatenate([ref[ep]['push'], ref[ep]['settle']])))
        tp, fp, fn = tp + a, fp + b, fn + c
    micro = 2 * tp / max(2 * tp + fp + fn, 1)
    print(f'[random, EVENT-count-matched] TP={tp} FP={fp} FN={fn} '
          f'micro F1={micro:.3f}')

    # ---- 3. density comparison ----------------------------------------
    print('=' * 70)
    print('3. PELT density vs event density (per episode)')
    cps_per, ev_per, seg_per = [], [], []
    for ep in hold_ids:
        d3 = np.load(SEG / f'ep{ep}.npz')
        cps_per.append(len(d3['cps_pelt']))
        ev_per.append(len(ref[ep]['push']) + len(ref[ep]['settle']))
        seg_per.append(len(d3['cps_pelt']) + 1)
    print(f'PELT cps/ep:  median={np.median(cps_per):.0f} '
          f'mean={np.mean(cps_per):.2f}')
    print(f'events/ep:     median={np.median(ev_per):.0f} '
          f'mean={np.mean(ev_per):.2f}')
    print(f'segments/ep (=cps+1): median={np.median(seg_per):.0f}')
    both = [(c, e) for c, e in zip(cps_per, ev_per)]
    fewer = sum(1 for c, e in both if c < e)
    print(f'episodes where PELT cps < events: {fewer}/'
          f'{len(hold_ids)}')

    # ---- 4. FP feature profile ----------------------------------------
    print('=' * 70)
    print('4. FP feature profile vs matched events vs all rows')
    all_rows = {'f1': [], 'f2': [], 'f3': []}
    for ep in hold_ids:
        fd = np.load(FEAT / f'ep{ep}.npz')
        all_rows['f1'].append(fd['f1'][1:])
        all_rows['f2'].append(fd['f2'][2:])
        all_rows['f3'].append(fd['f3'][3:])
    all_rows = {k: np.concatenate(v) for k, v in all_rows.items()}
    tp_f = np.array([[x for x in row if np.isfinite(x)] for row in tp_feat])
    fp_f = np.array([[x for x in row if np.isfinite(x)] for row in fp_feat])
    print(f'{"":10} {"f1":>8} {"f2":>8} {"f3":>8}   n')
    print(f'{"TP events":10} '
          f'{np.nanmedian(tp_f[:, 0]):>8.2f} '
          f'{np.nanmedian(tp_f[:, 1]):>8.2f} '
          f'{np.nanmedian(tp_f[:, 2]):>8.2f}   {len(tp_f)}')
    print(f'{"FP cps":10} '
          f'{np.nanmedian(fp_f[:, 0]):>8.2f} '
          f'{np.nanmedian(fp_f[:, 1]):>8.2f} '
          f'{np.nanmedian(fp_f[:, 2]):>8.2f}   {len(fp_f)}')
    print(f'{"all rows":10} '
          f'{np.nanmedian(all_rows["f1"]):>8.2f} '
          f'{np.nanmedian(all_rows["f2"]):>8.2f} '
          f'{np.nanmedian(all_rows["f3"]):>8.2f}   {len(all_rows["f1"])}')

    # ---- 5. FN temporal position ---------------------------------------
    print('=' * 70)
    print('5. FN position within episode (thirds)')
    for t in ('push', 'settle'):
        pos = np.array(fn_pos[t])
        if len(pos) == 0:
            continue
        thirds = [((pos < 1 / 3).sum(), ((pos >= 1 / 3) & (pos < 2 / 3)
                                         ).sum(), (pos >= 2 / 3).sum())]
        print(f'{t:6} FN n={len(pos)}: first1/3={thirds[0][0]} '
              f'mid1/3={thirds[0][1]} last1/3={thirds[0][2]}')


if __name__ == '__main__':
    main()
