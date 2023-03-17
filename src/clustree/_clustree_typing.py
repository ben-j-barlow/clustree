from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import matplotlib as mpl
import pandas as pd

OUTPUT_PATH_TYPE = Optional[Union[str, Path]]

NODE_CONFIG_TYPE = [
    int,  # (K, k) hashed
    dict[str, Any],  # 'color': XYZ, 'samples': 75, 'image': np.array
]

EDGE_CONFIG_TYPE = [
    int,  # (K, k_start, k_end) hashed
    dict[str, Any],  # 'color', 'samples', 'alpha', 'start', 'end', 'res'
]

DATA_INPUT_TYPE = Union[str, Path, pd.DataFrame]
IMAGE_INPUT_TYPE = Union[str, Path]
ORIENTATION_INPUT_TYPE = Literal["vertical", "horizontal"]
MIN_CLUSTER_NUMBER_TYPE = Literal[0, 1]
CIRCLE_POS_TYPE = Optional[Literal["tl", "t", "tr", "l", "r", "bl", "b", "br"]]

NODE_COLOR_TYPE = str  # e.g. 'samples', 'K', data col name
EDGE_COLOR_TYPE = str
COLOR_AGG_TYPE = Optional[Union[Callable, str]]
CMAP_TYPE = Union[mpl.colors.Colormap, str]
