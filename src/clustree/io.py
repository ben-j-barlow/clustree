from collections import defaultdict
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from clustree.clustree_typing import IMAGE_CONFIG_TYPE


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
    """
    Read list of files from a given directory using plt.imread.

    :param to_read: List[str], file names to read, show include file extension
    :param path: str or pathlib.Path, directory containing images.
    :param errors: bool, whether to raise error or include blank image if K_k \
    configuration absent from directory.
    :return: dict, key takes 'K_k' format, value is loaded image. Image could be fake.
    """
    if isinstance(path, str):
        path = Path(path)
    to_return = defaultdict(dict)

    for file in to_read:
        k_upper = file[0]
        k_lower = file[2]
        try:
            to_return[k_upper][k_lower] = plt.imread(Path(path / file))
        except FileNotFoundError as err:
            if errors:
                raise err
            to_return[k_upper][k_lower] = get_fake_img(k_upper=k_upper, k_lower=k_lower)
    return to_return
