#TODO: switch to use TYPE_CHECKING
from collections import defaultdict
from pathlib import Path
from typing import Any, Union
import numpy as np

NODE_CONFIG_TYPE = [
    int, # K
    dict[
        int, # k
        dict[str, Any] # 'color': XYZ, 'samples': 75, 'image': np.array
    ]
]

EDGE_CONFIG_TYPE = [
    int, # (K, k) hashed
    dict[str, Any] # 'color', 'samples', 'alpha', 'start', 'end', 'res'
]

#TODO: change to int by changing code throughout repo
IMAGE_CONFIG_TYPE = defaultdict[str, dict[str, np.ndarray]]

IMAGE_INPUT_TYPE = Union[str, Path, dict[str, np.ndarray]]
