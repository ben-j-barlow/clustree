import re
from pathlib import Path
from typing import List

import pandas as pd

from clustree.clustree_typing import (
    DATA_INPUT_TYPE,
    IMAGE_CONFIG_TYPE,
    IMAGE_INPUT_TYPE,
)
from clustree.io import read_images


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


def handle_images(images: IMAGE_INPUT_TYPE, errors: bool, kk: int) -> IMAGE_CONFIG_TYPE:
    if isinstance(images, (str, Path)):
        return read_images(
            to_read=[f"{ele}.png" for ele in get_img_name_pattern(kk=kk)],
            path=images,
            errors=errors,
        )
    return images


def handle_data(data: DATA_INPUT_TYPE) -> pd.DataFrame:
    if isinstance(data, (str, Path)):
        return pd.read_csv(data)
    return data
