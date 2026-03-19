from typing import Dict
import torch
import numpy as np
import copy
import sys
sys.path.append('/home/icrlab/imitation/improved_dp/diffusion_policy')
from common.utils import dict_apply
from common.replay_buffer import ReplayBuffer
from common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from .base_dataset import BaseDataset

class ManiSkillDataset(BaseDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'pmp_xy_plane', 'pmp_xz_plane', 'xy_plane', 'xz_plane',]) # 'left_img', 'right_img'
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
            'agent_pos': self.replay_buffer['state'][..., :],
            'xy_plane': self.replay_buffer['xy_plane'],
            'xz_plane': self.replay_buffer['xz_plane'],
            # 'yz_plane': self.replay_buffer['yz_plane'],
            # 'rgb': self.replay_buffer['rgb'],
            'pmp_xy_plane': self.replay_buffer['pmp_xy_plane'],
            'pmp_xz_plane': self.replay_buffer['pmp_xz_plane'],
            # 'pmp_yz_plane': self.replay_buffer['pmp_yz_plane'],
        }
        normalizer = LinearNormalizer() # [-1,1]
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:, ].astype(np.float32)
        # point_cloud = sample['point_cloud'][:, ].astype(np.float32)
        # rgb = sample['rgb'][:, ].astype(np.uint8)
        xy_plane = sample['xy_plane'][:, ].astype(np.uint8)
        xz_plane = sample['xz_plane'][:, ].astype(np.uint8)
        # yz_plane = sample['yz_plane'][:, ].astype(np.uint8)
        
        data = {
            'obs': {
                # 'point_cloud': point_cloud,
                'agent_pos': agent_pos,
                # 'rgb': rgb,
                'xy_plane': xy_plane,
                # 'yz_plane': yz_plane,
                'xz_plane': xz_plane,
                'pmp_xy_plane': sample['pmp_xy_plane'],
                'pmp_xz_plane': sample['pmp_xz_plane'],
                # 'pmp_yz_plane': sample['pmp_yz_plane'],
                # 'left_img': sample['left_img'][:, ].
            },
            'action': sample['action'].astype(np.float32)
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data