from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from clustree._clustree_typing import IMAGE_CONFIG_TYPE
from clustree._hash import hash_node_id


def get_fake_img(k_upper: str, k_lower: str, w: int = 40, h: int = 40) -> np.ndarray:
    """
    Create fake image reading 'K_k'.
    :param k_upper: cluster resolution
    :param k_lower: cluster number
    :param w: number of pixels
    :param h: number of pixels
    :return: image with white background and black text
    """
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    out = f"{k_upper}_{k_lower}"
    _, _, i, j = draw.textbbox((0, 0), out, font=font)
    draw.text(((w - i) / 2, (h - j) / 2), out, font=font, fill="black")
    return np.asarray(img)


def read_images(
    to_read: List[str], path: Union[str, Path], errors: bool = True
) -> IMAGE_CONFIG_TYPE:
    if isinstance(path, str):
        path = Path(path)
    to_return = dict()

    for file in to_read:
        k_upper, k_lower = tuple(file.split(".")[0].split("_"))
        node_id = hash_node_id(k_upper=int(k_upper), k_lower=int(k_lower))
        try:
            to_return[node_id] = plt.imread(Path(path / file))
        except FileNotFoundError as err:
            if errors:
                raise err
            to_return[node_id] = get_fake_img(k_upper=file[0], k_lower=file[2])
    return to_return
