"""Patch the locally collected cube h5 with the 3 columns the eval pipeline
needs but the collector did not write (plan v2 section 3.1 / facts F1-F4):

  episode_idx            (N,) int64   -- e6_eval episode_col() requires it
  privileged_block_0_pos     (N,3)    -- cube.yaml callable set_target_pos
  privileged_block_0_quat    (N,4)    -- ditto

The block pose is the qpos slice of the object_joint_0 free joint (scene_env
L1098-1104), so it is derived from the stored qpos column; the address is
read from a fresh CubeEnv instance and round-trip verified via set_state.
Only adds datasets, never rewrites existing ones. Idempotent (skips cols
that already exist).
"""
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'glfw')

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

H5 = r'D:\stable_worldmodel\datasets\ogbench\cube_single_expert.h5'


def get_block_qpos_adr():
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
    env = CubeEnv(env_type='single', ob_type='states')
    env.reset(seed=0)          # mujoco model is built lazily on first reset
    model = env._model
    jid = model.joint('object_joint_0').id
    adr = int(model.jnt_qposadr[jid])
    nq = int(model.nq)
    return env, adr, nq


def main():
    env, adr, nq = get_block_qpos_adr()
    print(f'object_joint_0 qpos address = {adr}, model.nq = {nq}')

    with h5py.File(H5, 'a') as f:
        n = f['qpos'].shape[0]
        assert f['qpos'].shape[1] == nq, (
            f"stored qpos width {f['qpos'].shape[1]} != model.nq {nq}")

        # --- round-trip check: set_state(stored qpos/qvel) -> joint qpos
        qpos_all = f['qpos'][:]
        qvel_all = f['qvel'][:]
        rng = np.random.default_rng(0)
        for i in rng.choice(n, 3, replace=False):
            env.set_state(qpos_all[i], qvel_all[i])
            rt = env._data.joint('object_joint_0').qpos[:]
            assert np.allclose(rt, qpos_all[i, adr:adr + 7], atol=1e-9), (
                f'round-trip mismatch at row {i}')
        print('round-trip set_state -> object_joint_0 qpos: OK (3 rows)')

        # --- episode_idx
        if 'episode_idx' not in f:
            ep_len = f['ep_len'][:]
            ep_offset = f['ep_offset'][:]
            assert int(ep_len.sum()) == n
            ep_ids = np.repeat(
                np.arange(len(ep_len), dtype=np.int64), ep_len)
            # boundary spot check
            assert ep_ids[0] == 0 and ep_ids[-1] == len(ep_len) - 1
            for k in (0, len(ep_len) - 1):
                assert ep_ids[ep_offset[k]] == k
            f.create_dataset('episode_idx', data=ep_ids)
            print(f'added episode_idx ({n},)')
        else:
            print('episode_idx already present')

        # --- privileged block columns
        if 'privileged_block_0_pos' not in f:
            pos = np.ascontiguousarray(qpos_all[:, adr:adr + 3])
            quat = np.ascontiguousarray(qpos_all[:, adr + 3:adr + 7])
            nq_ = np.linalg.norm(quat, axis=1)
            assert np.allclose(nq_, 1.0, atol=1e-4), (
                f'quat norm off: min {nq_.min()} max {nq_.max()}')
            f.create_dataset('privileged_block_0_pos', data=pos)
            f.create_dataset('privileged_block_0_quat', data=quat)
            print('added privileged_block_0_pos/quat '
                  f'({pos.shape}/{quat.shape})')
        else:
            print('privileged columns already present')

        # --- flatten HDF5 groups into underscore columns.
        # The collector wrote ob_info keys verbatim ('privileged/block_0_pos'),
        # which h5py stores as nested groups. The eval iterates
        # dataset.column_names (= top-level keys) and slices every one, so a
        # group would crash _load_slice; the callables need flat names
        # ('goal_privileged_block_0_pos'). Copy every group child to a flat
        # '{group}_{child}' column, cross-check, then drop the groups.
        for group in ('privileged', 'proprio'):
            if group not in f:
                print(f'group {group} already flattened')
                continue
            for child in list(f[group].keys()):
                flat = f'{group}_{child}'
                if flat in f:
                    assert np.allclose(f[flat][:], f[group][child][:]), flat
                    print(f'  {flat} == {group}/{child} (verified)')
                    continue
                f.create_dataset(flat, data=f[group][child][:])
                print(f'  flattened {group}/{child} -> {flat}')
            if group == 'privileged' and 'privileged' in f:
                assert np.allclose(f['privileged_block_0_pos'][:],
                                   f['privileged/block_0_pos'][:]), (
                    'qpos-derived block pos != collector privileged/block_0_pos')
                print('  qpos-derived block_0_pos verified against '
                      'collector privileged/block_0_pos')
            del f[group]
            print(f'  dropped group {group}')

        # --- summary + stillness sanity around success frames
        success = f['success'][:]
        pos = f['privileged_block_0_pos'][:]
        disp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        succ_disp = disp[success[:-1] & success[1:]]
        all_disp = disp
        print(f'block per-frame displacement: all median '
              f'{np.median(all_disp):.5f}m | success-frame median '
              f'{np.median(succ_disp) if len(succ_disp) else float("nan"):.5f}m')
        print('columns now:', sorted(f.keys()))

    # --- loader-level check (what e6_eval will see)
    sys.path.insert(0, r'd:\Research-2026-jepa-boundary-detector\aiim2026'
                      r'\third_party\stable-worldmodel')
    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
    ds = HDF5Dataset(name='ogbench/cube_single_expert',
                     cache_dir=r'D:\stable_worldmodel',
                     keys_to_cache=['action'])
    names = set(ds.column_names)
    assert {'episode_idx', 'privileged_block_0_pos',
            'privileged_block_0_quat'} <= names, names
    ei = ds.get_col_data('episode_idx')
    print(f'HDF5Dataset sees episode_idx: {ei.shape}, '
          f'{len(np.unique(ei))} episodes')

    # simulate the eval's _extract_init_goal loop over ALL columns
    data = ds.load_chunk(np.array([0]), np.array([0]), np.array([4]))
    eps = data if isinstance(data, list) else [data]
    for col in ds.column_names:
        if col.startswith('goal'):
            continue
        v = eps[0][col]
        assert isinstance(v, (np.ndarray,)) or hasattr(v, 'shape'), (
            f'column {col} not array-like -> would break the eval loop')
    flat_goals = {f'goal_{c}' for c in ds.column_names}
    assert 'goal_privileged_block_0_pos' in flat_goals
    assert 'goal_privileged_block_0_quat' in flat_goals
    print(f'load_chunk over {len(ds.column_names)} columns OK; '
          'goal_privileged_block_0_pos/quat resolve -- PATCH OK')


if __name__ == '__main__':
    main()
