from pathlib import Path

import pytest
from PIL import Image

from clustree.io import read_images


def test_read_images():
    # Test case 1: kk = 1, path to existing directory, errors = True
    path = "tests/test_data/"
    assert read_images(1, path) == {"1_1": Image.open(Path(path + "1_1.png"))}

    # Test case 2: kk = 2, path to non-existing directory, errors = True
    with pytest.raises(FileNotFoundError):
        read_images(2, path="abcd", errors=True)

    # Test case 3: kk = 2, path to existing directory, errors = False
    assert read_images(2, path, errors=False) == {
        "1_1": Image.open(Path(path + "1_1.png")),
        "2_1": Image.open(Path(path + "2_1.png")),
        "2_2": Image.open(Path(path + "2_2.png")),
    }

    # Test case 4: kk = 0, path to existing directory, errors = True
    with pytest.raises(ValueError):
        read_images(0, path, errors="raise")
        read_images(0, path, errors="ignore")

    # Test case 6: kk = 2, pathlib.Path object, errors = True
    assert read_images(2, Path(path)) == {
        "1_1": Image.open(path + "1_1.png"),
        "2_1": Image.open(path + "2_1.png"),
        "2_2": Image.open(path + "2_2.png"),
    }
