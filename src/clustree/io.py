from pathlib import Path
from typing import Union

from PIL import Image


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
