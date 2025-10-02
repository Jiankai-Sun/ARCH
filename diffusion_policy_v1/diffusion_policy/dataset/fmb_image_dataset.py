from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class FMBImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            input_type='v3',
            state_dim=10,
            file_name_spec=None,  # ['_M_S_'],
            input_padding='ones',
            selected_primitive=None,  # ['grasp'],
            remapping=False,
            level=None,
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_dir(
            zarr_path, keys=['img', 'state', 'action', 'trajectory_id'], input_type=input_type, state_dim=state_dim,
            file_name_spec=file_name_spec, input_padding=input_padding, selected_primitive=selected_primitive, remapping=remapping,
            level=level)
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'], #[...,:2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        agent_pos = sample['state'].astype(np.float32)  # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32), # T, action_dim
            'trajectory_id': sample['trajectory_id'],
            'primitive_id': sample['primitive_id'],
        }
        # print('sample.keys(): ', sample.keys())
        # print('sample[action].shape: ', sample['action'].shape)  # [16, 7]
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        # print(0, data.keys())
        trajectory_id = data.pop('trajectory_id').tolist()
        # print(data.keys(), trajectory_id)
        # dict_keys(['obs', 'action'])
        torch_data = dict_apply(data, torch.from_numpy)
        # print(1, 'torch_data.keys(): ', torch_data.keys())
        # print('torch_data', torch_data)
        # print(len(trajectory_id), data['action'].shape)
        return torch_data, trajectory_id


def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = FMBImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
