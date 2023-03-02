from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Union

import matplotlib as mpl
import numpy as np
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

# TODO: change to int by changing code throughout repo
IMAGE_CONFIG_TYPE = defaultdict[str, dict[str, np.ndarray]]

DATA_INPUT_TYPE = Union[str, Path, pd.DataFrame]
IMAGE_INPUT_TYPE = Union[str, Path, dict[str, np.ndarray]]

# TODO: improve typing for node_color
NODE_COLOR_TYPE = Any  # e.g. 'samples', 'K', data col name
EDGE_COLOR_TYPE = Any
COLOR_AGG_TYPE = Optional[Union[Callable, str]]
CMAP_TYPE = Optional[Union[mpl.colors.Colormap, str]]
