"""Acceptance #3: cross-machine task-set consistency (4060 vs mac).

Mac's E1 flat runs (eval_wm.py) printed the sampled row indices and the
valid-starting-point count into runs/flat_d{d}_seed{s}.log. e6_eval.py's
sampling is verbatim from eval_wm.py, so replicating the draw on the 4060
h5 and matching all six (d, seed) arrays proves the paired design carries
over (same seed -> same tasks on both machines).
"""
import re
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

H5 = r'D:\stable_worldmodel\datasets\pusht_expert_train.h5'
RUNS = Path(__file__).resolve().parent.parent / 'runs'
NUM_EVAL = 50  # mac pilot n


def load_cols():
    with h5py.File(H5, 'r') as f:
        ep = f['episode_idx'][:]
        st = f['step_idx'][:]
        ep_len_col = f['ep_len'][:]
    return ep, st, ep_len_col


def sample(ep, st, ep_len_col, goal_offset, seed):
    # e6_eval: episode_len = max(step_idx | ep) + 1, computed per unique ep
    assert np.all(np.diff(ep) >= 0), 'episode_idx not row-sorted'
    uniq, starts = np.unique(ep, return_index=True)
    ep_len = np.maximum.reduceat(st, starts) + 1
    assert np.array_equal(ep_len, ep_len_col), 'ep_len column mismatch'
    max_start_idx = ep_len - goal_offset - 1
    row_ep = np.searchsorted(uniq, ep)
    valid_mask = st <= max_start_idx[row_ep]
    valid_indices = np.nonzero(valid_mask)[0]
    g = np.random.default_rng(seed)
    r = g.choice(len(valid_indices), size=NUM_EVAL, replace=False)
    return int(valid_mask.sum()), np.sort(valid_indices[r])


def parse_log(path):
    txt = path.read_text()
    n_valid = None
    m = re.search(r'(\d+) valid starting points', txt)
    if m:
        n_valid = int(m.group(1))
    m2 = re.search(r'\[\s*((?:\d[\d\s]*)?)\]', txt, re.S)
    arr = np.array([int(x) for x in m2.group(1).split()], dtype=np.int64)
    return n_valid, arr


def main():
    ep, st, ep_len_col = load_cols()
    print(f'rows={len(ep)} episodes={len(ep_len_col)} '
          f'total_steps={int(ep_len_col.sum())}')
    all_ok = True
    for d in (25, 50, 75):
        for seed in (42, 43, 44):
            log = RUNS / f'flat_d{d}_seed{seed}.log'
            n_mac, arr_mac = parse_log(log)
            n_loc, arr_loc = sample(ep, st, ep_len_col, d, seed)
            pool_ok = (n_mac is None) or (n_mac == n_loc)
            if len(arr_mac) == NUM_EVAL:
                arr_ok = np.array_equal(arr_mac, arr_loc)
            else:  # truncated log: compare the printed prefix
                arr_ok = np.array_equal(arr_mac, arr_loc[:len(arr_mac)])
            ok = pool_ok and arr_ok
            all_ok &= ok
            print(f'd{d} s{seed}: pool mac={n_mac} local={n_loc} '
                  f'({"OK" if pool_ok else "MISMATCH"}); '
                  f'indices[{len(arr_mac)}] '
                  f'{"IDENTICAL" if arr_ok else "MISMATCH"} '
                  f'-> {"PASS" if ok else "FAIL"}')
    print('VERDICT:', 'PASS - task sets identical across machines'
          if all_ok else 'FAIL - investigate before Task A')


if __name__ == '__main__':
    main()
