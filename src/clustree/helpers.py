from pathlib import Path
from typing import Union

from PIL import Image, ImageDraw, ImageFont


def _get_img_name_pattern(kk: int) -> list[str]:
    """
    Given kk, produce list of form 'K_k'. \
    For example, kk = 2 produces ['1_1', '2_1', '2_2'].

    :param kk: int, highest cluster resolution, 1 or greater.
    :return: list, containing K_k combinations
    """
    return [f"{K}_{k}" for K in range(1, kk + 1) for k in range(1, K + 1)]


def _get_fake_img(k_upper: str, k_lower: str) -> Image:
    W, H = 40, 40
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    out = f"{k_upper}_{k_lower}"
    _, _, w, h = draw.textbbox((0, 0), out, font=font)
    draw.text(((W - w) / 2, (H - h) / 2), out, font=font, fill="black")
    return img


def read_images(
    kk: int, path: Union[str, Path], errors: bool = True
) -> dict[str, Image]:
    """
    Given kk, read images named 'K_k.png' for 1 <= k <= K, 1 <= K <= kk. Can include real or fake images.

    :param kk: int, highest cluster resolution to read.
    :param path: str or pathlib.Path, directory containing images.
    :param errors: bool, whether to raise error or include blank image if K_k configuration absent from directory.
    :return: dict, key takes 'K_k' format, value is loaded image. Image could be fake.
    """
    if kk < 1:
        raise ValueError(f"cluster resolution should be greater than 0, not {str(kk)}")
    if isinstance(path, str):
        path = Path(path)
    to_read = _get_img_name_pattern(kk=kk)
    to_return = {}

    for file in to_read:
        try:
            to_return[file] = Image.open(Path(path / f"{file}.png"))
        except FileNotFoundError as err:
            if errors:
                raise err
            to_return[file] = _get_fake_img(k_upper=file[0], k_lower=file[2])
    return to_return
