import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from clustree._clustree_typing import (
    DATA_INPUT_TYPE,
    IMAGE_CONFIG_TYPE,
    IMAGE_INPUT_TYPE,
)
from clustree._io import read_images


def get_img_name_pattern(kk: int, start_at_1: bool) -> list[str]:
    if start_at_1:
        return [f"{K}_{k}" for K in range(1, kk + 1) for k in range(1, K + 1)]
    return [f"{K}_{k}" for K in range(1, kk + 1) for k in range(0, K)]


def get_and_check_cluster_cols(cols: List[str], prefix: str, user_kk: Optional[int]) -> int:
    pattern = f"{prefix}[0-9]+"
    if user_kk:
        cols = [x for x in cols if re.match(pattern, x) and int(x.removeprefix(prefix)) <= user_kk]
    else:
        cols = [x for x in cols if re.match(pattern, x)]
    cols_as_int = [int(ele.removeprefix(prefix)) for ele in cols]

    if max(cols_as_int) != len(cols_as_int):
        raise ValueError(
            f"cols with prefix '{prefix}' should be consecutive integers: \
         '{prefix}1', '{prefix}2', ..., '{prefix}N'"
        )
    return max(cols_as_int)


def handle_images(
    images: IMAGE_INPUT_TYPE, errors: bool, kk: int, start_at_1: bool
) -> IMAGE_CONFIG_TYPE:
    if isinstance(images, (str, Path)):
        return read_images(
            to_read=[
                f"{ele}.png"
                for ele in get_img_name_pattern(kk=kk, start_at_1=start_at_1)
            ],
            path=images,
            errors=errors,
        )
    return images


def handle_data(data: DATA_INPUT_TYPE) -> pd.DataFrame:
    if isinstance(data, (str, Path)):
        return pd.read_csv(data)
    return data
