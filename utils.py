import datetime
from typing import List, Tuple

import numpy as np
from pandas import DataFrame


class AverageMeter:
    def __init__(self):
        self.value = None
        self.sum = None
        self.count = 0
        self.average = None

    def update(self, data, count: int = 1):
        self.value = data
        if self.sum is None:
            self.sum = data * count
        else:
            self.sum += data * count
        self.count += count
        self.average = self.sum / self.count

    def reset(self):
        self.value, self.sum, self.count, self.average = None, None, 0, None


def find_highly_correlated(corr: DataFrame, threshold: float = 0.9) -> List[Tuple[str, str]]:
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=1)] = True
    val = np.abs(corr.to_numpy()) * mask
    keys = list(corr.columns)
    idx = np.fliplr(np.unravel_index(np.argsort(val, axis=None), val.shape))
    num_high_pairs = (val >= threshold).sum()
    pairs = []
    for i in range(num_high_pairs):
        if idx[0, i] == 0 or idx[1, i] == 0:
            continue
        pairs.append((keys[idx[0, i]], keys[idx[1, i]]))
    return pairs


def find_least_correlated_with(pairs: List[Tuple[str, str]], corr: DataFrame, target: str) -> List[str]:
    assert target in corr.columns, f'Target key should exists in dataframe, got {target}.'
    must_delete = []
    for pair in pairs:
        if abs(corr.loc[target, pair[0]]) < abs(corr.loc[target, pair[1]]):
            must_delete.append(pair[0])
        else:
            must_delete.append(pair[1])
    return must_delete


def ordinal(n: int) -> str:
    return "%d%s" % (n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])


def time_between(df: DataFrame, key: str = 'transaction_time_utc') -> np.ndarray:
    time_points = df.loc[:, key].values
    tp = np.zeros_like(time_points)
    for i, t in enumerate(time_points):
        tp[i] = datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    tp = np.sort(tp)
    time_bet = np.ediff1d(tp)
    for i, t in enumerate(time_bet):
        time_bet[i] = t.total_seconds()
    return time_bet


def count_active_days(df: DataFrame, key: str = 'transaction_time_utc') -> int:
    time_points = df.loc[:, key].values
    tp = np.zeros_like(time_points)
    for i, t in enumerate(time_points):
        tp[i] = ''.join(t.split(' ')[0].split('-'))
    return len(np.unique(tp))
