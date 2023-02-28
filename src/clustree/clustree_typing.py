from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional, Union

import matplotlib as mpl
import numpy as np

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

IMAGE_INPUT_TYPE = Union[str, Path, dict[str, np.ndarray]]

# TODO: improve typing for node_color
NODE_COLOR_TYPE = Any  # Union[str, mpl.colors]  # e.g. 'samples', 'K', data col name
NODE_COLOR_AGG_TYPE = Optional[Callable]
NODE_CMAP_TYPE = Optional[mpl.colors.Colormap]