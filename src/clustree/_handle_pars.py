import re
import typing
from collections import defaultdict
from pathlib import Path
from typing import List, Union

import numpy as np

from clustree.io import read_images

if typing.TYPE_CHECKING:
    from pandas import DataFrame

IMAGES_TYPE = Union[str, Path, dict[str, np.ndarray]]


def get_img_name_pattern(kk: int) -> list[str]:
    """
    Given kk, produce list of form 'K_k'. \
    For example, kk = 2 produces ['1_1', '2_1', '2_2'].

    :param kk: int, highest cluster resolution, 1 or greater.
    :return: list, containing K_k combinations
    """
    return [f"{K}_{k}" for K in range(1, kk + 1) for k in range(1, K + 1)]


def get_and_check_cluster_cols(cols: List[str], prefix: str) -> tuple[List[str], int]:
    """

    :param cols: column names to search for prefix in
    :param prefix: used to identify cluster membership columns
    :return: cols (not sorted 1 to N)
    """
    pattern = f"{prefix}[0-9]+"
    cols = [x for x in cols if re.match(pattern, x)]
    cols_as_int = [int(ele.removeprefix(prefix)) for ele in cols]

    if max(cols_as_int) != len(cols_as_int):
        raise ValueError(
            f"cols with prefix '{prefix}' should be consecutive integers: \
         '{prefix}1', '{prefix}2', ..., '{prefix}N'"
        )
    return cols, max(cols_as_int)


def to_k_k(
    k_upper: int, k_lower: Union[int, str, List[int], List[str]]
) -> Union[str, List[str]]:
    """
    Convert from k as int to 'K_k' as str .
    """
    if isinstance(k_lower, (int, str)):
        return f"{str(k_upper)}_{str(k_lower)}"
    return [f"{str(k_upper)}_{str(ele)}" for ele in k_lower]


def from_k_k(k_k: Union[str, List[str]]) -> Union[int, List[int]]:
    """
    Convert from 'K_k' as str format to k as int.
    """
    if isinstance(k_k, str):
        return int(k_k[2])
    return [int(ele[2]) for ele in k_k]


def append_k_k_cols(data: "DataFrame", prefix: str, kk: int) -> "DataFrame":
    """
    Use cluster membership columns in data to produce cluster membership in form 'K_k'.
    """
    for k_upper in range(1, kk + 1):
        k_upper_str = str(k_upper)
        if f"{k_upper_str}_k" not in data.columns:
            data.insert(
                loc=data.shape[1],
                column=f"{k_upper_str}_k",
                value=to_k_k(k_upper=k_upper, k_lower=data[f"{prefix}{k_upper_str}"]),
            )
    return data


def handle_images(
    images: IMAGES_TYPE, errors: bool, kk: int
) -> Union[None, defaultdict[str, dict[str, np.ndarray]]]:
    if isinstance(images, (str, Path)):
        return read_images(
            to_read=[f"{ele}.png" for ele in get_img_name_pattern(kk=kk)],
            path=images,
            errors=errors,
        )

    # isinstance(images, dict)
    _images = defaultdict(dict)
    for key, img in images.items():
        _images[key[0]][key[2]] = img
    return _images
