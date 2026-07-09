from typing import Dict
import sys
import torch
import torch.nn

sys.path.append('/home/icrlab/imitation/improved_dp/diffusion_policy')
from model.common.normalizer import LinearNormalizer
from diffusion_policy.common.utils import dict_apply

class BaseDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseDataset':
        # return an empty dataset by default
        return BaseDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs:
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()
