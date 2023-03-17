import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from clustree._clustree_typing import DATA_INPUT_TYPE


def get_and_check_cluster_cols(
    cols: List[str], prefix: str, user_kk: Optional[int]
) -> int:
    pattern = f"{prefix}[0-9]+"
    if user_kk:
        cols = [
            x
            for x in cols
            if re.match(pattern, x) and int(x.removeprefix(prefix)) <= user_kk
        ]
    else:
        cols = [x for x in cols if re.match(pattern, x)]
    cols_as_int = [int(ele.removeprefix(prefix)) for ele in cols]

    if max(cols_as_int) != len(cols_as_int):
        raise ValueError(
            f"cols with prefix '{prefix}' should be consecutive integers: \
         '{prefix}1', '{prefix}2', ..., '{prefix}N'"
        )
    return max(cols_as_int)


def handle_data(data: DATA_INPUT_TYPE) -> pd.DataFrame:
    if isinstance(data, (str, Path)):
        return pd.read_csv(data)
    return data
