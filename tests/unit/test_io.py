from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from clustree.io import read_images
from tests.helpers import INPUT_DIR


def test_read_images():
    # Test case 1: path to existing directory, errors = True
    path = INPUT_DIR
    out = read_images(to_read=["1_1.png"], path=path)
    exp = {"1": {"1": plt.imread(Path(path + "1_1.png"))}}
    assert out.keys() == exp.keys()
    assert out["1"].keys() == exp["1"].keys()
    assert np.array_equal(out["1"]["1"], exp["1"]["1"])

    # Test case 2: path to non-existing directory
    with pytest.raises(FileNotFoundError):
        read_images(to_read=["1_1"], path="abcd", errors=True)
        # errors = False but still expect error since errors param controls .png
        # not found, not directory not found
        read_images(to_read=["1_1"], path="abcd", errors=False)

    # Test case 3: path to existing directory, errors = False
    out = read_images(
        to_read=["1_1.png", "2_1.png", "2_2.png"], path=path, errors=False
    )
    exp = {
        "1": {"1": plt.imread(Path(path + "1_1.png"))},
        "2": {
            "1": plt.imread(Path(path + "2_1.png")),
            "2": plt.imread(Path(path + "2_2.png")),
        },
    }
    assert out.keys() == exp.keys()
    assert out["1"].keys() == exp["1"].keys()
    assert out["2"].keys() == exp["2"].keys()
    assert np.array_equal(out["1"]["1"], exp["1"]["1"])
    assert np.array_equal(out["2"]["1"], exp["2"]["1"])
    assert np.array_equal(out["2"]["2"], exp["2"]["2"])

    # Test case 4: pathlib.Path object, errors = True
    out = read_images(to_read=["1_1.png", "2_1.png", "2_2.png"], path=Path(path))
    assert out.keys() == exp.keys()
    assert out["1"].keys() == exp["1"].keys()
    assert out["2"].keys() == exp["2"].keys()
    assert np.array_equal(out["1"]["1"], exp["1"]["1"])
    assert np.array_equal(out["2"]["1"], exp["2"]["1"])
    assert np.array_equal(out["2"]["2"], exp["2"]["2"])
