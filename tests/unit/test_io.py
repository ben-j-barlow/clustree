from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from clustree.hash import hash_node_id
from clustree.io import read_images
from tests.helpers import INPUT_DIR


def test_read_images():
    # Test case 1: path to existing directory, errors = True
    path = INPUT_DIR
    node_id = hash_node_id(1, 1)

    out = read_images(to_read=["1_1.png"], path=path)
    exp = {node_id: plt.imread(Path(path + "1_1.png"))}
    assert out.keys() == exp.keys()
    assert np.array_equal(out[node_id], exp[node_id])

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
        hash_node_id(1, 1): plt.imread(Path(path + "1_1.png")),
        hash_node_id(2, 1): plt.imread(Path(path + "2_1.png")),
        hash_node_id(2, 2): plt.imread(Path(path + "2_2.png")),
    }
    assert out.keys() == exp.keys()

    keys = list(out.keys())
    assert np.array_equal(out[keys[0]], exp[keys[0]])
    assert np.array_equal(out[keys[1]], exp[keys[1]])
    assert np.array_equal(out[keys[2]], exp[keys[2]])

    # Test case 4: pathlib.Path object, errors = True
    out = read_images(to_read=["1_1.png", "2_1.png", "2_2.png"], path=Path(path))
    assert out.keys() == exp.keys()
    assert np.array_equal(out[keys[0]], exp[keys[0]])
    assert np.array_equal(out[keys[1]], exp[keys[1]])
    assert np.array_equal(out[keys[2]], exp[keys[2]])
