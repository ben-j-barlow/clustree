from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from clustree.graph import clustree
from tests.helpers import INPUT_DIR, OUTPUT_DIR

DATA_FILES = pytest.mark.datafiles(
    INPUT_DIR + "1_1.png",
    INPUT_DIR + "2_1.png",
    INPUT_DIR + "2_2.png",
    INPUT_DIR + "3_1.png",
    INPUT_DIR + "3_2.png",
    INPUT_DIR + "3_3.png",
    INPUT_DIR + "iris.csv",
)


@DATA_FILES
def test_clustree(datafiles):
    to_read = [Path(ele) for ele in datafiles.listdir()]
    data_files = {
        file.stem: pd.read_csv(file) for file in to_read if file.suffix == ".csv"
    }
    iris = data_files["iris"]

    dg = clustree(data=iris, prefix="K", images=INPUT_DIR, draw=False, path=None)

    assert dg.number_of_edges() == 6
    assert dg.number_of_nodes() == 6
    assert set(dg.edges) == {
        ("1_1", "2_1"),
        ("1_1", "2_2"),
        ("2_1", "3_1"),
        ("2_1", "3_2"),
        ("2_2", "3_2"),
        ("2_2", "3_3"),
    }


@DATA_FILES
def test_clustree_draw(datafiles):
    to_read = [Path(ele) for ele in datafiles.listdir()]
    img_files = {
        file.stem: plt.imread(file) for file in to_read if file.suffix == ".png"
    }
    data_files = {
        file.stem: pd.read_csv(file) for file in to_read if file.suffix == ".csv"
    }
    iris = data_files["iris"]

    # images as dict
    clustree(
        data=iris,
        prefix="K",
        images=img_files,
        draw=True,
        path=OUTPUT_DIR + "test_dict.png",
    )

    # images as str
    clustree(
        data=iris,
        prefix="K",
        images=INPUT_DIR,
        draw=True,
        path=OUTPUT_DIR + "test_str.png",
    )

    # images as path
    clustree(
        data=iris,
        prefix="K",
        images=Path(INPUT_DIR),
        draw=True,
        path=OUTPUT_DIR + "test_path.png",
    )
