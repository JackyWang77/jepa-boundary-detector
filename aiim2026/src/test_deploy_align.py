"""Train-serving alignment regression (hard gate for the E6 boundary arm).

Replays E2 episodes through BOTH pipelines and asserts element-wise
agreement:
  A. 18-channel feature rows: online BoundaryPolicy._features vs the
     offline R2 pipeline (raw f1/f2/f3 -> causal multi-scale stats ->
     z-score with r2_meta mu/sd). Gate: max |diff| < 1e-5.
  B. head probabilities: online causal CNN prefix passes (position m)
     vs offline full-sequence forward. Gate: max |diff| < 1e-5.
Exits non-zero on failure. Run: python test_deploy_align.py [ep_id]
"""

import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary_policy import BoundaryPolicy
from e4_train_head import CausalCNN
import stable_worldmodel as swm

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / 'features'
H5 = Path.home() / '.stable_worldmodel/datasets/pusht_expert_train.h5'
FS, GATE = 5, 1e-5
N_TEST = 3


def main():
    r2m = json.load(open(ROOT / 'models' / 'r2_meta.json'))
    feats_meta = json.load(open(FEAT / 'meta.json'))
    ep_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None

    model = swm.wm.utils.load_pretrained('pusht/weights.pt').eval()
    BP = BoundaryPolicy.__new__(BoundaryPolicy)
    BP.dev = 'cpu'
    BP.a_mean = torch.tensor(feats_meta['action_mean'])
    BP.a_std = torch.tensor(feats_meta['action_std_plus_eps'])
    BP.model = model
    BP.windows = tuple(dict.fromkeys(
        int(c.split('_W')[1]) for c in r2m['channels']))
    BP.feat_mu = np.asarray(r2m['feat_mu'], np.float32)
    BP.feat_sd = np.asarray(r2m['feat_sd'], np.float32)
    BP.head_class = r2m['head_class']
    BP.head = CausalCNN(in_ch=len(r2m['channels']))
    BP.head.load_state_dict(
        torch.load(r2m['head_ckpt'], map_location='cpu'))
    BP.head.eval()

    ids = ([ep_arg] if ep_arg
           else [feats_meta['episode_ids'][i]
                 for i in range(N_TEST)])
    worst_feat = worst_p = 0.0
    with h5py.File(H5, 'r', swmr=True) as f5:
        for ep in ids:
            fd = np.load(FEAT / f'ep{ep}.npz')
            K = int(fd['model_len'])
            z, raw = (fd['z'],
                      f5['action'][
                          int(f5['ep_offset'][ep]):
                          int(f5['ep_offset'][ep]) + K * FS])

            # -- offline rows (R2 pipeline) --------------------------
            rows_off, mu_off = [], None
            from e2_extract_features import causal_stats
            chans = []
            for name in ('f1', 'f2', 'f3'):
                for w in BP.windows:
                    m, sd = causal_stats(fd[name], w)
                    chans.extend([m, sd])
            X = np.stack(chans, axis=1)[3:]
            X = ((X - BP.feat_mu) / BP.feat_sd).astype(np.float32)

            # -- online replay ----------------------------------------
            st = BP._new_state()
            p_online = {}
            for m in range(K):
                s, row = BP._features(
                    st, torch.from_numpy(z[m]).float())
                if row is not None:
                    d = float(np.abs(row - X[s - 3]).max())
                    worst_feat = max(worst_feat, d)
                    p_online[s] = BP._head_step(st, row)
                if m < K - 1:
                    for a in raw[m * FS:(m + 1) * FS]:
                        st['acts'].append(
                            np.asarray(a, np.float32).reshape(-1))

            with torch.no_grad():
                p_off = torch.sigmoid(BP.head(
                    torch.from_numpy(X)[None]))[0].numpy()
            for s, p in p_online.items():
                worst_p = max(worst_p, abs(p - float(p_off[s - 3])))

            print(f'ep{ep}: K={K} rows_online={len(p_online)} '
                  f'worst_feat={worst_feat:.2e} worst_p={worst_p:.2e}')

    ok = worst_feat < GATE and worst_p < GATE
    print(f'\nHARD GATE (<{GATE}): feature rows {worst_feat:.2e}, '
          f'head probs {worst_p:.2e} -> '
          f'{"PASS" if ok else "FAIL"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
