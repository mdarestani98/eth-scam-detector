from abc import ABC
from collections.abc import Callable
from typing import Tuple, Any, Union, List, Dict

import networkx as nx
import numpy as np
import pandas as pd
import torch
import tqdm
from pandas import DataFrame
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset

from utils import time_between, count_active_days


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


keys = ['avg_time_between_sent_eth', 'std_time_between_sent_eth', 'avg_time_between_rec_eth',
        'std_time_between_rec_eth', 'no_active_days', 'activity_time', 'no_sent_eth', 'no_rec_eth',
        'unique_sent_address_eth', 'unique_rec_addresses_eth', 'min_value_sent_eth', 'max_value_sent_eth',
        'avg_value_sent_eth', 'std_value_sent_eth', 'min_value_rec_eth', 'max_value_rec_eth', 'avg_value_rec_eth',
        'std_value_rec_eth', 'total_gas_sent_eth', 'std_gas_sent_eth', 'total_gas_rec_eth', 'std_gas_rec_eth',
        'avg_gas_price_sent_eth', 'std_gas_price_sent_eth', 'avg_gas_price_rec_eth', 'std_gas_price_rec_eth',
        'avg_time_between_sent_tkn', 'avg_time_between_rec_tkn', 'no_sent_tkn', 'no_rec_tkn', 'unique_sent_address_tkn',
        'unique_rec_addresses_tkn', 'total_gas_sent_tkn', 'std_gas_sent_tkn', 'total_gas_rec_tkn', 'std_gas_rec_tkn',
        'avg_gas_price_sent_tkn', 'std_gas_price_sent_tkn', 'avg_gas_price_rec_tkn', 'std_gas_price_rec_tkn'
        ]

general_keys = [k for k in keys if not any([kk in k for kk in ['eth', 'tkn']])]
eth_keys = [k for k in keys if 'eth' in k]
tkn_keys = [k for k in keys if 'tkn' in k]

trx_keys = ['from_account', 'to_account', 'transaction_time_utc', 'value', 'gas', 'gas_price']


def extract_features(g: nx.MultiGraph, acc_list: Union[List[str], np.ndarray]) -> DataFrame:
    features = DataFrame(columns=['account'] + keys)
    progress = tqdm.tqdm(acc_list, total=len(acc_list), ncols=150, smoothing=0.9,
                         desc='Extracting features for all accounts')
    for acc in progress:
        progress.set_description(f'Extracting features for account {acc}')
        trx = nx.to_pandas_edgelist(g, source='from_account', target='to_account', nodelist=[acc])
        sent = trx[trx.from_account == acc]
        rec = trx[trx.to_account == acc]
        sent_eth = sent[sent.value > 0]
        sent_tkn = sent[sent.value == 0]
        rec_eth = rec[rec.value > 0]
        rec_tkn = rec[rec.value == 0]
        all_time_bet = time_between(trx)
        sent_eth_time_bet = time_between(sent_eth)
        rec_eth_time_bet = time_between(rec_eth)
        sent_tkn_time_bet = time_between(sent_tkn)
        rec_tkn_time_bet = time_between(rec_tkn)
        row = {
            'account': acc,
            'avg_time_between_sent_eth': sent_eth_time_bet.mean() if len(sent_eth) > 1 else np.nan,
            'std_time_between_sent_eth': sent_eth_time_bet.std() if len(sent_eth) > 1 else np.nan,
            'avg_time_between_rec_eth': rec_eth_time_bet.mean() if len(rec_eth) > 1 else np.nan,
            'std_time_between_rec_eth': rec_eth_time_bet.std() if len(rec_eth) > 1 else np.nan,
            'no_sent_eth': len(sent_eth),
            'no_rec_eth': len(rec_eth),
            'unique_sent_address_eth': len(sent_eth.to_account.unique()),
            'unique_rec_addresses_eth': len(rec_eth.to_account.unique()),
            'min_value_sent_eth': sent_eth.value.values.min() if len(sent_eth) > 0 else np.nan,
            'max_value_sent_eth': sent_eth.value.values.max() if len(sent_eth) > 0 else np.nan,
            'avg_value_sent_eth': sent_eth.value.values.mean() if len(sent_eth) > 0 else np.nan,
            'std_value_sent_eth': sent_eth.value.values.std() if len(sent_eth) > 0 else np.nan,
            'min_value_rec_eth': rec_eth.value.values.min() if len(rec_eth) > 0 else np.nan,
            'max_value_rec_eth': rec_eth.value.values.max() if len(rec_eth) > 0 else np.nan,
            'avg_value_rec_eth': rec_eth.value.values.mean() if len(rec_eth) > 0 else np.nan,
            'std_value_rec_eth': rec_eth.value.values.std() if len(rec_eth) > 0 else np.nan,
            'total_gas_sent_eth': (sent_eth.gas.values * sent_eth.gas_price.values).sum(),
            'std_gas_sent_eth': np.nanstd(sent_eth.gas.values * sent_eth.gas_price.values),
            'total_gas_rec_eth': (rec_eth.gas.values * rec_eth.gas_price.values).sum(),
            'std_gas_rec_eth': np.nanstd(rec_eth.gas.values * rec_eth.gas_price.values),
            'avg_gas_price_sent_eth': sent_eth.gas_price.mean(),
            'std_gas_price_sent_eth': sent_eth.gas_price.std(),
            'avg_gas_price_rec_eth': rec_eth.gas_price.mean(),
            'std_gas_price_rec_eth': rec_eth.gas_price.std(),
            'avg_time_between_sent_tkn': sent_tkn_time_bet.mean() if len(sent_tkn) > 1 else np.nan,
            'std_time_between_sent_tkn': sent_tkn_time_bet.std() if len(sent_tkn) > 1 else np.nan,
            'avg_time_between_rec_tkn': rec_tkn_time_bet.mean() if len(rec_tkn) > 1 else np.nan,
            'std_time_between_rec_tkn': rec_tkn_time_bet.std() if len(rec_tkn) > 1 else np.nan,
            'no_sent_tkn': len(sent_tkn),
            'no_rec_tkn': len(rec_tkn),
            'unique_sent_address_tkn': len(sent_tkn.to_account.unique()),
            'unique_rec_addresses_tkn': len(rec_tkn.to_account.unique()),
            'total_gas_sent_tkn': (sent_tkn.gas.values * sent_tkn.gas_price.values).sum(),
            'std_gas_sent_tkn': np.nanstd(sent_tkn.gas.values * sent_tkn.gas_price.values),
            'total_gas_rec_tkn': (rec_tkn.gas.values * rec_tkn.gas_price.values).sum(),
            'std_gas_rec_tkn': np.nanstd(rec_tkn.gas.values * rec_tkn.gas_price.values),
            'avg_gas_price_sent_tkn': sent_tkn.gas_price.mean(),
            'std_gas_price_sent_tkn': sent_tkn.gas_price.std(),
            'avg_gas_price_rec_tkn': rec_tkn.gas_price.mean(),
            'std_gas_price_rec_tkn': rec_tkn.gas_price.std(),
            'activity_time': all_time_bet.sum(),
            'no_active_days': count_active_days(trx)
        }
        features = pd.concat([features, DataFrame([row])], ignore_index=True)
    progress.close()
    return features


def aggregate_neighbors_features(df: DataFrame):
    row = {f'{k}_agg': np.nanmean(df[k].values) for k in df.columns if k not in ['account']}
    return row
