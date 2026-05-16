from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
import copy
from functools import cached_property
import cv2

def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0

def rechunk_recompress_array(group, name, 
        chunks=None, chunk_length=None,
        compressor=None, tmp_key='_temp'):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)
    
    if compressor is None:
        compressor = old_arr.compressor
    
    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr

primitive_name_remapping_rule_v0 = {'move_up': 'grasp', 'place_on_fixture': 'regrasp', 'go_to_board': 'insert',}
primitive_name_remapping_rule = {'move_up': 'grasp', 'regrasp': 'grasp', 'go_to_board': 'insert',}
primitive_name_idx_remapping = {'move_up': 0, 'grasp': 1, 'place_on_fixture': 2, 'regrasp': 3, 'go_to_board': 4, 'insert':5,}
primitive_name_idx_remapping = {
                                    "AssemblyReset": 0,
                                    "AssemblyScriptedGrasp": 1,
                                    "AssemblyMoveToFixture": 2,
                                    "AssemblyScriptedPlace": 3,
                                    "AssemblyScriptedRegrasp": 4,
                                    "AssemblyMoveToBoardFromFixture": 5,
                                    "AssemblyScriptedInsert": 6,
                                    "AssemblyInsert": 7,
                                    "AssemblyMoveToBoardFromGrasp": 8,
                                    "AssemblyScriptedGraspHorizontal": 9,
                                    "AssemblyMoveToFixtureHorizontal": 10,
                                    "AssemblyScriptedPlaceHorizontal": 11,
                                }

def get_optimal_chunks(shape, dtype, 
        target_chunk_bytes=2e6, 
        max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape)-1
    for i in range(len(shape)-1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[:i+1])
        if this_chunk_bytes <= target_chunk_bytes \
            and next_chunk_bytes > target_chunk_bytes:
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(this_max_chunk_length, math.ceil(
            target_chunk_bytes / item_chunk_bytes))
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


class ReplayBuffer:
    """
    Zarr-based temporal datastructure.
    Assumes first dimension to be time. Only chunk in time dimension.
    """
    def __init__(self, 
            root: Union[zarr.Group, 
            Dict[str,dict]]):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        """
        assert('data' in root)
        assert('meta' in root)
        assert('episode_ends' in root['meta'])
        for key, value in root['data'].items():
            # if key in ['trajectory_id']:
            #     continue
            # print(key, value.shape, root['meta']['episode_ends'])
            assert(value.shape[0] == root['meta']['episode_ends'][-1]), 'value.shape: {}, root[meta][episode_ends][-1]: {}'.format(value.shape, root['meta']['episode_ends'][-1])
        self.root = root

    # ============= create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        data = root.require_group('data', overwrite=False)
        meta = root.require_group('meta', overwrite=False)
        if 'episode_ends' not in meta:
            episode_ends = meta.zeros('episode_ends', shape=(0,), dtype=np.int64,
                compressor=None, overwrite=False)
        return cls(root=root)
    
    @classmethod
    def create_empty_numpy(cls):
        root = {
            'data': dict(),
            'meta': {
                'episode_ends': np.zeros((0,), dtype=np.int64)
            }
        }
        return cls(root=root)
    
    @classmethod
    def create_from_group(cls, group, **kwargs):
        if 'data' not in group:
            # create from stratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exist
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode='r', **kwargs):
        """
        Open a on-disk zarr directly (for dataset larger than memory).
        Slower.
        """
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        return cls.create_from_group(group, **kwargs)
    
    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(cls, src_store, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Load to memory.
        """
        src_root = zarr.group(src_store)
        root = None
        if store is None:
            # numpy backend
            meta = dict()
            for key, value in src_root['meta'].items():
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]

            if keys is None:
                keys = src_root['data'].keys()
            data = dict()
            for key in keys:
                arr = src_root['data'][key]
                data[key] = arr[:]

            root = {
                'meta': meta,
                'data': data
            }
        else:
            root = zarr.group(store=store)
            # copy without recompression
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(source=src_store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
            data_group = root.create_group('data', overwrite=True)
            if keys is None:
                keys = src_root['data'].keys()
            for key in keys:
                value = src_root['data'][key]
                cks = cls._resolve_array_chunks(
                    chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value)
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store, dest=store,
                        source_path=this_path, dest_path=this_path,
                        if_exists=if_exists
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
        buffer = cls(root=root)
        return buffer

    @classmethod
    def copy_from_dir(cls, src_store, store=None, keys=None,
                        chunks: Dict[str, tuple] = dict(),
                        compressors: Union[dict, str, numcodecs.abc.Codec] = dict(),
                        if_exists='replace', input_type='v3', state_dim=10, num_seq=20,
                        file_name_spec='_M_S_', input_padding='ones', selected_primitive=None,
                        remapping=False, level=None,
                        **kwargs):
        """
        Load to memory.
        """
        # src_root = zarr.group(src_store)
        episode_ends = 0
        meta = dict()
        meta['episode_ends'] = []
        meta['trajectory_name'] = []
        data = dict()
        data['img'] = []
        data['state'] = []
        data['action'] = []
        data['trajectory_id'] = []
        data['primitive_id'] = []
        all_files = sorted(i for i in os.listdir(src_store) if '.npy' in i)
        downsampling_index = np.linspace(0, 256 - 1, 96, dtype=int)
        x_indices, y_indices = np.meshgrid(downsampling_index, downsampling_index, indexing='ij')
        all_primitives = set()
        num_trajectories = 0
        for file_i, each_file in enumerate(all_files):
            satisfy_file_name_spec_flag = False
            for each_file_name_spec in file_name_spec:
                if each_file_name_spec in each_file:
                    satisfy_file_name_spec_flag = True
            if not satisfy_file_name_spec_flag:
                continue
            full_path = os.path.join(src_store, each_file)
            traj = np.load(full_path, allow_pickle=True)
            # for k, v in traj.item().items():
            #     length = len(traj.item()[k])
            #     print(k, length)
            primitive = traj.item()['primitive']
            # print('primitive: ', primitive)
            if remapping:
                primitive = [primitive_name_remapping_rule.get(prim, prim) for prim in primitive]
            all_primitives.update(primitive)
            # print('primitive: ', primitive)
            # print('all_primitives:', all_primitives)
            # print('selected_primitive: ', selected_primitive)
            if selected_primitive:
                start_indices = []
                end_indices = []
                for each_selected_primitive in selected_primitive:
                    in_segment = False
                    for i, action in enumerate(primitive):
                        if action == each_selected_primitive and not in_segment:
                            start_indices.append(i)
                            in_segment = True
                        elif (action != each_selected_primitive and in_segment) or i == len(primitive) - 1:
                            end_indices.append(i)
                            in_segment = False
                # print('start_indices, end_indices: ', start_indices, end_indices, primitive)
                for segment_i, (start_index, end_index) in enumerate(zip(start_indices, end_indices)):
                    # print(segment_i, (start_index, end_index), episode_ends)
                    # data, meta, episode_ends = slice_data(data, meta, traj, episode_ends, each_file, segment_i, start_index, end_index, x_indices, y_indices, input_type, input_padding)
                    episode_ends = episode_ends + (end_index - start_index)
                    meta['episode_ends'].append(episode_ends)
                    trajectory_name = each_file + '_{:03d}_{:04d}_{:04d}'.format(segment_i, start_index, end_index)
                    meta['trajectory_name'].append(trajectory_name)
                    print('each_file: ', trajectory_name)
                    # data[each_file] = {}
                    img_list = []
                    if 'obs/side_1' in input_type:
                        side_1 = traj.item()['obs/side_1'] #[start_index: end_index]
                        img_list.append(side_1)
                    elif not input_padding == 'ignore':  # 'obs/side_1' not in input_type:
                        side_1 = traj.item()['obs/side_1'] #[start_index: end_index]
                        if input_padding == 'ones':
                            img_list.append(np.ones_like(side_1))
                        elif input_padding == 'zeros':
                            img_list.append(np.zeros_like(side_1))

                    if 'obs/side_1_depth' in input_type:
                        # print(1, traj.item()['obs/side_1_depth'][..., None].shape)
                        side_1_depth = traj.item()['obs/side_1_depth'][..., None] # [start_index: end_index]
                        img_list.append(side_1_depth)
                    elif not input_padding == 'ignore':
                        side_1_depth = traj.item()['obs/side_1_depth'][..., None] # [start_index: end_index]
                        if input_padding == 'ones':
                            img_list.append(np.ones_like(side_1_depth))
                        elif input_padding == 'zeros':
                            img_list.append(np.zeros_like(side_1_depth))

                    if 'obs/side_2' in input_type:
                        side_2 = traj.item()['obs/side_2'] #[start_index: end_index]
                        img_list.append(side_2)
                    elif not input_padding == 'ignore':
                        side_2 = traj.item()['obs/side_2'] # [start_index: end_index]
                        if input_padding == 'ones':
                            img_list.append(np.ones_like(side_2))
                        elif input_padding == 'zeros':
                            img_list.append(np.zeros_like(side_2))

                    if 'obs/side_2_depth' in input_type:
                        side_2_depth = traj.item()['obs/side_2_depth'][..., None] # [start_index: end_index]
                        img_list.append(side_2_depth)
                    elif not input_padding == 'ignore':
                        side_2_depth = traj.item()['obs/side_2_depth'][..., None] # [start_index: end_index]
                        if input_padding == 'ones':
                            img_list.append(np.ones_like(side_2_depth))
                        elif input_padding == 'zeros':
                            img_list.append(np.zeros_like(side_2_depth))

                    if 'obs/wrist_1' in input_type:
                        wrist_1 = traj.item()['obs/wrist_1'] # [start_index: end_index]
                        img_list.append(wrist_1)
                    elif not input_padding == 'ignore':
                        wrist_1 = traj.item()['obs/wrist_1'] # [start_index: end_index]
                        if input_padding == 'ones':
                            img_list.append(np.ones_like(wrist_1))
                        elif input_padding == 'zeros':
                            img_list.append(np.zeros_like(wrist_1))

                    if 'obs/wrist_1_depth' in input_type:
                        wrist_1_depth = traj.item()['obs/wrist_1_depth'][..., None] # [start_index: end_index]
                        img_list.append(wrist_1_depth)
                    elif not input_padding == 'ignore':  # 'obs/side_1' not in input_type:
                        wrist_1_depth = traj.item()['obs/wrist_1_depth'][..., None] # [start_index: end_index]
                        if input_padding == 'ones':
                            img_list.append(np.ones_like(wrist_1_depth))
                        elif input_padding == 'zeros':
                            img_list.append(np.zeros_like(wrist_1_depth))

                    if 'obs/wrist_2' in input_type:
                        wrist_2 = traj.item()['obs/wrist_2']  # [start_index: end_index]
                        img_list.append(wrist_2)
                    elif not input_padding == 'ignore':  # 'obs/side_1' not in input_type:
                        wrist_2 = traj.item()['obs/wrist_2']  # [start_index: end_index]
                        if input_padding == 'ones':
                            img_list.append(np.ones_like(wrist_2))
                        elif input_padding == 'zeros':
                            img_list.append(np.zeros_like(wrist_2))

                    if 'obs/wrist_2_depth' in input_type:
                        wrist_2_depth = traj.item()['obs/wrist_2_depth'][..., None]  # [start_index: end_index]
                        img_list.append(wrist_2_depth)
                    elif not input_padding == 'ignore':  # 'obs/side_1' not in input_type:
                        wrist_2_depth = traj.item()['obs/wrist_2_depth'][..., None] # [start_index: end_index]
                        if input_padding == 'ones':
                            img_list.append(np.ones_like(wrist_2_depth))
                        elif input_padding == 'zeros':
                            img_list.append(np.zeros_like(wrist_2_depth))

                    # print('img.shape: ', img.shape)  # img.shape:  (204, 256, 256, 3)
                    # for each_img in img:
                    #     resized_img = cv2.resize(each_img, dsize=(96, 96))
                    #     data['img'].append(resized_img)
                    # print('resized_img.shape: ', resized_img.shape)

                    # print('downsampling_index: ', downsampling_index)
                    img_np = np.concatenate(img_list, axis=-1)
                    downsampled_img = img_np[:, x_indices, y_indices]  # (N, 96, 96, 16)
                    downsampled_img = downsampled_img[start_index:end_index]
                    # print('downsampled_img.shape: ', downsampled_img.shape)
                    data['img'].append(downsampled_img)
                    state_list = []
                    # print(traj.item()['primitive'], len(traj.item()['primitive']), len(traj.item()['obs/tcp_pose']))
                    if 'obs/tcp_pose' in input_type:
                        tcp_pose = traj.item()['obs/tcp_pose']  # [start_index: end_index]  # (N, 7)
                        state_list.append(tcp_pose)
                    elif not input_padding == 'ignore':  # 'obs/tcp_pose' not in input_type:
                        tcp_pose = traj.item()['obs/tcp_pose']  # [start_index: end_index]
                        if input_padding == 'ones':
                            state_list.append(np.ones_like(tcp_pose))
                        elif input_padding == 'zeros':
                            state_list.append(np.zeros_like(tcp_pose))

                    if 'obs/tcp_vel' in input_type:
                        tcp_vel = traj.item()['obs/tcp_vel'] # [start_index: end_index]  # (N, 6)
                        state_list.append(tcp_vel)
                    elif not input_padding == 'ignore':  # 'obs/tcp_vec' not in input_type:
                        tcp_vel = traj.item()['obs/tcp_vel'] # [start_index: end_index]
                        if input_padding == 'ones':
                            state_list.append(np.ones_like(tcp_vel))
                        elif input_padding == 'zeros':
                            state_list.append(np.zeros_like(tcp_vel))

                    if 'obs/tcp_force' in input_type:
                        tcp_force = traj.item()['obs/tcp_force'] # [start_index: end_index]  # (N, 3)
                        state_list.append(tcp_force)
                    elif not input_padding == 'ignore':  # 'obs/tcp_vec' not in input_type:
                        tcp_force = traj.item()['obs/tcp_force'] # [start_index: end_index]
                        if input_padding == 'ones':
                            state_list.append(np.ones_like(tcp_force))
                        elif input_padding == 'zeros':
                            state_list.append(np.zeros_like(tcp_force))

                    if 'obs/tcp_torque' in input_type:
                        tcp_torque = traj.item()['obs/tcp_torque'] # [start_index: end_index]  # (N, 3)
                        state_list.append(tcp_torque)
                    elif not input_padding == 'ignore':
                        tcp_torque = traj.item()['obs/tcp_torque'] # [start_index: end_index]
                        if input_padding == 'ones':
                            state_list.append(np.ones_like(tcp_torque))
                        elif input_padding == 'zeros':
                            state_list.append(np.zeros_like(tcp_torque))

                    if 'obs/gripper_pose' in input_type:
                        gripper_pose = traj.item()['obs/gripper_pose'][..., None] # [start_index: end_index]  # (N,)
                        state_list.append(gripper_pose)
                    elif not input_padding == 'ignore':
                        gripper_pose = traj.item()['obs/gripper_pose'][..., None] # [start_index: end_index]
                        if input_padding == 'ones':
                            state_list.append(np.ones_like(gripper_pose))
                        elif input_padding == 'zeros':
                            state_list.append(np.zeros_like(gripper_pose))

                    q = traj.item()['obs/q'] # [start_index: end_index]  # (N, 7)
                    dq = traj.item()['obs/dq'] # [start_index: end_index]  # (N, 7)
                    jacobian = traj.item()['obs/jacobian'][..., None] # [start_index: end_index]  # (N, 6, 7)
                    state_np = np.concatenate(state_list, axis=-1)  # (7 + 6 + 3 + 3 + 1) = 20
                    state_np = state_np[start_index:end_index]
                    # print('state_np.shape: ', state_np.shape, )
                    data['state'].append(state_np)
                    action = traj.item()['actions']
                    action = action[start_index:end_index]
                    # print('action.shape: ', action.shape, )
                    data['action'].append(action) # [start_index: end_index])
                    primitive_ids = [primitive_name_idx_remapping[p] for p in primitive[start_index:end_index]]
                    data['primitive_id'].append(primitive_ids)
                    for i in range(len(traj.item()['primitive'][start_index: end_index])):  #
                        data['trajectory_id'].append(
                            '{}_{:03d}_{:04d}_{:04d}_{:04d}'.format(each_file, segment_i, start_index, end_index, i))
                    # print('len: ', len(data['state']), data['state'][0].shape, data['img'][0].shape, start_index, end_index)
            else:
                # data, meta, episode_ends = slice_data(data, meta, traj, episode_ends, each_file, 0, 0,
                #                                       None, x_indices, y_indices, input_type, input_padding)
                episode_ends = episode_ends + len(traj.item()['primitive'])
                meta['episode_ends'].append(episode_ends)
                meta['trajectory_name'].append(each_file)
                print('each_file: ', each_file)
                # data[each_file] = {}
                img_list = []
                if 'obs/side_1' in input_type:
                    side_1 = traj.item()['obs/side_1']
                    img_list.append(side_1)
                elif not input_padding == 'ignore': # 'obs/side_1' not in input_type:
                    side_1 = traj.item()['obs/side_1']
                    if input_padding == 'ones':
                        img_list.append(np.ones_like(side_1))
                    elif input_padding == 'zeros':
                        img_list.append(np.zeros_like(side_1))

                if 'obs/side_1_depth' in input_type:
                    side_1_depth = traj.item()['obs/side_1_depth'][..., None]
                    img_list.append(side_1_depth)
                elif not input_padding == 'ignore':
                    side_1_depth = traj.item()['obs/side_1_depth'][..., None]
                    if input_padding == 'ones':
                        img_list.append(np.ones_like(side_1_depth))
                    elif input_padding == 'zeros':
                        img_list.append(np.zeros_like(side_1_depth))

                if 'obs/side_2' in input_type:
                    side_2 = traj.item()['obs/side_2']
                    img_list.append(side_2)
                elif not input_padding == 'ignore':
                    side_2 = traj.item()['obs/side_2']
                    if input_padding == 'ones':
                        img_list.append(np.ones_like(side_2))
                    elif input_padding == 'zeros':
                        img_list.append(np.zeros_like(side_2))

                if 'obs/side_2_depth' in input_type:
                    side_2_depth = traj.item()['obs/side_2_depth'][..., None]
                    img_list.append(side_2_depth)
                elif not input_padding == 'ignore':
                    side_2_depth = traj.item()['obs/side_2_depth'][..., None]
                    if input_padding == 'ones':
                        img_list.append(np.ones_like(side_2_depth))
                    elif input_padding == 'zeros':
                        img_list.append(np.zeros_like(side_2_depth))

                if 'obs/wrist_1' in input_type:
                    wrist_1 = traj.item()['obs/wrist_1']
                    img_list.append(wrist_1)
                elif not input_padding == 'ignore':
                    wrist_1 = traj.item()['obs/wrist_1']
                    if input_padding == 'ones':
                        img_list.append(np.ones_like(wrist_1))
                    elif input_padding == 'zeros':
                        img_list.append(np.zeros_like(wrist_1))

                if 'obs/wrist_1_depth' in input_type:
                    wrist_1_depth = traj.item()['obs/wrist_1_depth'][..., None]
                    img_list.append(wrist_1_depth)
                elif not input_padding == 'ignore': # 'obs/side_1' not in input_type:
                    wrist_1_depth = traj.item()['obs/wrist_1_depth'][..., None]
                    if input_padding == 'ones':
                        img_list.append(np.ones_like(wrist_1_depth))
                    elif input_padding == 'zeros':
                        img_list.append(np.zeros_like(wrist_1_depth))

                if 'obs/wrist_2' in input_type:
                    wrist_2 = traj.item()['obs/wrist_2']
                    img_list.append(wrist_2)
                elif not input_padding == 'ignore': # 'obs/side_1' not in input_type:
                    wrist_2 = traj.item()['obs/wrist_2']
                    if input_padding == 'ones':
                        img_list.append(np.ones_like(wrist_2))
                    elif input_padding == 'zeros':
                        img_list.append(np.zeros_like(wrist_2))

                if 'obs/wrist_2_depth' in input_type:
                    wrist_2_depth = traj.item()['obs/wrist_2_depth'][..., None]
                    img_list.append(wrist_2_depth)
                elif not input_padding == 'ignore': # 'obs/side_1' not in input_type:
                    wrist_2_depth = traj.item()['obs/wrist_2_depth'][..., None]
                    if input_padding == 'ones':
                        img_list.append(np.ones_like(wrist_2_depth))
                    elif input_padding == 'zeros':
                        img_list.append(np.zeros_like(wrist_2_depth))

                # print('img.shape: ', img.shape)  # img.shape:  (204, 256, 256, 3)
                # for each_img in img:
                #     resized_img = cv2.resize(each_img, dsize=(96, 96))
                #     data['img'].append(resized_img)
                    # print('resized_img.shape: ', resized_img.shape)

                # print('downsampling_index: ', downsampling_index)
                img_np = np.concatenate(img_list, axis=-1)
                downsampled_img = img_np[:, x_indices, y_indices]  # (N, 96, 96, 16)
                # print('downsampled_img.shape: ', downsampled_img.shape)
                data['img'].append(downsampled_img)
                state_list = []
                # print(traj.item()['primitive'], len(traj.item()['primitive']), len(traj.item()['obs/tcp_pose']))
                if 'obs/tcp_pose' in input_type:
                    tcp_pose = traj.item()['obs/tcp_pose']  # (N, 7)
                    state_list.append(tcp_pose)
                elif not input_padding == 'ignore': # 'obs/tcp_pose' not in input_type:
                    tcp_pose = traj.item()['obs/tcp_pose']
                    if input_padding == 'ones':
                        state_list.append(np.ones_like(tcp_pose))
                    elif input_padding == 'zeros':
                        state_list.append(np.zeros_like(tcp_pose))

                if 'obs/tcp_vel' in input_type:
                    tcp_vel = traj.item()['obs/tcp_vel'] # (N, 6)
                    state_list.append(tcp_vel)
                elif not input_padding == 'ignore': # 'obs/tcp_vec' not in input_type:
                    tcp_vel = traj.item()['obs/tcp_vel']
                    if input_padding == 'ones':
                        state_list.append(np.ones_like(tcp_vel))
                    elif input_padding == 'zeros':
                        state_list.append(np.zeros_like(tcp_vel))

                if 'obs/tcp_force' in input_type:
                    tcp_force = traj.item()['obs/tcp_force'] # (N, 3)
                    state_list.append(tcp_force)
                elif not input_padding == 'ignore': # 'obs/tcp_vec' not in input_type:
                    tcp_force = traj.item()['obs/tcp_force']
                    if input_padding == 'ones':
                        state_list.append(np.ones_like(tcp_force))
                    elif input_padding == 'zeros':
                        state_list.append(np.zeros_like(tcp_force))

                if 'obs/tcp_torque' in input_type:
                    tcp_torque = traj.item()['obs/tcp_torque'] # (N, 3)
                    state_list.append(tcp_torque)
                elif not input_padding == 'ignore':
                    tcp_torque = traj.item()['obs/tcp_torque']
                    if input_padding == 'ones':
                        state_list.append(np.ones_like(tcp_torque))
                    elif input_padding == 'zeros':
                        state_list.append(np.zeros_like(tcp_torque))

                if 'obs/gripper_pose' in input_type:
                    gripper_pose = traj.item()['obs/gripper_pose'][..., None] # (N,)
                    state_list.append(gripper_pose)
                elif not input_padding == 'ignore':
                    gripper_pose = traj.item()['obs/gripper_pose'][..., None]
                    if input_padding == 'ones':
                        state_list.append(np.ones_like(gripper_pose))
                    elif input_padding == 'zeros':
                        state_list.append(np.zeros_like(gripper_pose))

                q = traj.item()['obs/q']  # (N, 7)
                dq = traj.item()['obs/dq']   # (N, 7)
                jacobian = traj.item()['obs/jacobian'][..., None]  # (N, 6, 7)
                state_np = np.concatenate(state_list, axis=-1) # (7 + 6 + 3 + 3 + 1) = 20
                data['state'].append(state_np)
                data['action'].append(traj.item()['actions'])
                primitive_ids = [primitive_name_idx_remapping[p] for p in primitive]
                data['primitive_id'].append(primitive_ids)
                for i in range(len(traj.item()['primitive'])):
                    data['trajectory_id'].append('{}_{:03d}_{:04d}_{:04d}_{:04d}'.format(each_file, 0, 0, len(traj.item()['primitive']), i))
            num_trajectories = num_trajectories + 1
            # if file_i > num_seq:
            #     break
        # numpy backend

        # for key, value in src_root['meta'].items():
        #     if len(value.shape) == 0:
        #         meta[key] = np.array(value)
        #     else:
        #         meta[key] = value[:]

        # if keys is None:
        #     keys = src_root['data'].keys()

        # for key in keys:
        #     arr = src_root['data'][key]
        #     data[key] = arr[:]
        print('all_primitives: {}, selected_primitive: {}, # trajectories: {}'.format(all_primitives, selected_primitive, num_trajectories))
        for k, v in data.items():
            if k in ['trajectory_id']:
                data[k] = np.array(v)
                print('data: ', k, len(v))
                continue
            # print('data: ', k, len(v), v[0].shape)
            data[k] = np.concatenate(v, axis=0)
            print('data: ', k, data[k].shape)
        # for k, v in data.items():
        #     print('dict: ', k, len(v), v[0].shape)
        meta['episode_ends'] = np.asarray(meta['episode_ends'])
        root = {
            'meta': meta,
            'data': data
        }
        buffer = cls(root=root)
        return buffer
    
    @classmethod
    def copy_from_path(cls, zarr_path, backend=None, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Copy a on-disk zarr to in-memory compressed.
        Recommended
        """
        if backend == 'numpy':
            print('backend argument is deprecated!')
            store = None
        group = zarr.open(os.path.expanduser(zarr_path), 'r')
        return cls.copy_from_store(src_store=group.store, store=store, 
            keys=keys, chunks=chunks, compressors=compressors, 
            if_exists=if_exists, **kwargs)

    # ============= save methods ===============
    def save_to_store(self, store, 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(),
            if_exists='replace', 
            **kwargs):
        
        root = zarr.group(store)
        if self.backend == 'zarr':
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
        else:
            meta_group = root.create_group('meta', overwrite=True)
            # save meta, no chunking
            for key, value in self.root['meta'].items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape)
        
        # save data, chunk
        data_group = root.create_group('data', overwrite=True)
        for key, value in self.root['data'].items():
            cks = self._resolve_array_chunks(
                chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value)
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store, dest=store,
                        source_path=this_path, dest_path=this_path, if_exists=if_exists)
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
            else:
                # numpy
                _ = data_group.array(
                    name=key,
                    data=value,
                    chunks=cks,
                    compressor=cpr
                )
        return store

    def save_to_path(self, zarr_path,             
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(), 
            if_exists='replace', 
            **kwargs):
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(store, chunks=chunks, 
            compressors=compressors, if_exists=if_exists, **kwargs)

    @staticmethod
    def resolve_compressor(compressor='default'):
        if compressor == 'default':
            compressor = numcodecs.Blosc(cname='lz4', clevel=5, 
                shuffle=numcodecs.Blosc.NOSHUFFLE)
        elif compressor == 'disk':
            compressor = numcodecs.Blosc('zstd', clevel=5, 
                shuffle=numcodecs.Blosc.BITSHUFFLE)
        return compressor

    @classmethod
    def _resolve_array_compressor(cls, 
            compressors: Union[dict, str, numcodecs.abc.Codec], key, array):
        # allows compressor to be explicitly set to None
        cpr = 'nil'
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == 'nil':
            cpr = cls.resolve_compressor('default')
        return cpr
    
    @classmethod
    def _resolve_array_chunks(cls,
            chunks: Union[dict, tuple], key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks
    
    # ============= properties =================
    @cached_property
    def data(self):
        return self.root['data']
    
    @cached_property
    def meta(self):
        return self.root['meta']

    def update_meta(self, data):
        # sanitize data
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == 'zarr':
            for key, value in np_data.items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape,
                    overwrite=True)
        else:
            meta_group.update(np_data)
        
        return meta_group
    
    @property
    def episode_ends(self):
        return self.meta['episode_ends']
    
    def get_episode_idxs(self):
        import numba
        numba.jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result
        return _get_episode_idxs(self.episode_ends)
        
    
    @property
    def backend(self):
        backend = 'numpy'
        if isinstance(self.root, zarr.Group):
            backend = 'zarr'
        return backend
    
    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == 'zarr':
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]
    
    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def chunk_size(self):
        if self.backend == 'zarr':
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths

    def add_episode(self, 
            data: Dict[str, np.ndarray], 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict()):
        assert(len(data) > 0)
        is_zarr = (self.backend == 'zarr')

        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            assert(len(value.shape) >= 1)
            if episode_length is None:
                episode_length = len(value)
            else:
                assert(episode_length == len(value))
        new_len = curr_len + episode_length

        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            # create array
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value)
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value)
                    arr = self.data.zeros(name=key, 
                        shape=new_shape, 
                        chunks=cks,
                        dtype=value.dtype,
                        compressor=cpr)
                else:
                    # copy data to prevent modify
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                assert(value.shape[1:] == arr.shape[1:])
                # same method for both zarr and numpy
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            # copy data
            arr[-value.shape[0]:] = value
        
        # append to episode ends
        episode_ends = self.episode_ends
        if is_zarr:
            episode_ends.resize(episode_ends.shape[0] + 1)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len

        # rechunk
        if is_zarr:
            if episode_ends.chunks[0] < episode_ends.shape[0]:
                rechunk_recompress_array(self.meta, 'episode_ends', 
                    chunk_length=int(episode_ends.shape[0] * 1.5))
    
    def drop_episode(self):
        is_zarr = (self.backend == 'zarr')
        episode_ends = self.episode_ends[:].copy()
        assert(len(episode_ends) > 0)
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        for key, value in self.data.items():
            new_shape = (start_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)
        if is_zarr:
            self.episode_ends.resize(len(episode_ends)-1)
        else:
            self.episode_ends.resize(len(episode_ends)-1, refcheck=False)
    
    def pop_episode(self):
        assert(self.n_episodes > 0)
        episode = self.get_episode(self.n_episodes-1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result
    
    def get_episode_slice(self, idx):
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result
    
    # =========== chunking =============
    def get_chunks(self) -> dict:
        assert self.backend == 'zarr'
        chunks = dict()
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks
    
    def set_chunks(self, chunks: dict):
        assert self.backend == 'zarr'
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        assert self.backend == 'zarr'
        compressors = dict()
        for key, value in self.data.items():
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        assert self.backend == 'zarr'
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)
