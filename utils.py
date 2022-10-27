from typing import List, Tuple

import numpy as np
from pandas import DataFrame


def find_highly_correlated(corr: DataFrame, threshold: float = 0.9) -> List[Tuple[str, str]]:
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask, k=1)] = True
    val = np.abs(corr.to_numpy()) * mask
    keys = list(corr.columns)
    idx = np.fliplr(np.unravel_index(np.argsort(val, axis=None), val.shape))
    num_high_pairs = (val > threshold).sum()
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
