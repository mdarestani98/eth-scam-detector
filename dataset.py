from abc import ABC
from collections.abc import Callable
from typing import Tuple, Any, Union, List

import numpy as np
import torch
from pandas import DataFrame
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset


class Transform(nn.Module, ABC):
    def __init__(self) -> None:
        super(Transform, self).__init__()

    def __call__(self, features: Union[Tensor, np.ndarray], **kwargs) -> Union[Tensor, np.ndarray]:
        raise NotImplementedError


class Compose:
    def __init__(self, transform_list: List[Transform]) -> None:
        self.transforms = transform_list

    def __call__(self, features: Union[Tensor, np.ndarray], **kwargs) -> Union[Tensor, np.ndarray]:
        if type(features) is np.ndarray:
            res = np.copy(features)
        elif type(features) is Tensor:
            res = torch.clone(features)
        else:
            raise TypeError
        for t in self.transforms:
            res = t(res)
        return res


class ToTensor(Transform):
    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, features: np.ndarray, **kwargs) -> Tensor:
        features = torch.from_numpy(features)
        if not isinstance(features, torch.FloatTensor):
            features = features.float()
        return features


class Normalize(Transform):
    def __init__(self, mean: Union[List[float], Tuple[float, ...], np.ndarray],
                 std: Union[List[float], Tuple[float, ...], np.ndarray] = None, device: str = 'cpu') -> None:
        super(Normalize, self).__init__()
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        mean = np.array(mean)
        self.mean = torch.from_numpy(mean).view((-1,)).to(device)
        if std is not None:
            std = np.array(std)
            self.std = torch.from_numpy(std).view((-1,)).to(device)
        else:
            self.std = std

    def __call__(self, features: Tensor, **kwargs) -> Tensor:
        features = features.view((-1,))
        if self.std is None:
            features.sub_(self.mean)
        else:
            features.sub_(self.mean).div_(self.std)
        return features


class ETHDataset(Dataset):
    def __init__(self, df: DataFrame, transform: Callable[[np.ndarray], Tensor] = None, target_key: str = 'flag'):
        super(ETHDataset, self).__init__()
        self.df = df
        self.target_key = target_key
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Union[np.ndarray, Tensor], int]:
        row = self.df.iloc[idx].to_dict()
        target = row.pop(self.target_key)
        features = np.array(list(row.values()), dtype=float)
        if self.transform is not None:
            return self.transform(features), int(target)
        return features, int(target)
