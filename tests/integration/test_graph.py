from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from clustree.config import ClustreeConfig as cfg
from clustree.graph import clustree
from tests.helpers import INPUT_DIR, OUTPUT_DIR

IMG_FILES = pytest.mark.datafiles(
    INPUT_DIR + "1_1.png",
    INPUT_DIR + "2_1.png",
    INPUT_DIR + "2_2.png",
    INPUT_DIR + "3_1.png",
    INPUT_DIR + "3_2.png",
    INPUT_DIR + "3_3.png",
)


def test_clustree(datafiles, iris_data):
    dg = clustree(data=iris_data, prefix="K", images=INPUT_DIR, draw=False, path=None)

    assert dg.number_of_edges() == 6
    assert dg.number_of_nodes() == 6
    assert set(dg.edges) == {
        (cfg.hash_k_k(1, 1), cfg.hash_k_k(2, 1)),
        (cfg.hash_k_k(1, 1), cfg.hash_k_k(2, 2)),
        (cfg.hash_k_k(2, 1), cfg.hash_k_k(3, 1)),
        (cfg.hash_k_k(2, 1), cfg.hash_k_k(3, 2)),
        (cfg.hash_k_k(2, 2), cfg.hash_k_k(3, 2)),
        (cfg.hash_k_k(2, 2), cfg.hash_k_k(3, 3)),
    }


@IMG_FILES
def test_clustree_draw(datafiles, iris_data):
    to_read = [Path(ele) for ele in datafiles.listdir()]
    img_files = {file.stem: plt.imread(file) for file in to_read}

    # images as dict
    clustree(
        data=iris_data,
        prefix="K",
        images=img_files,
        draw=True,
        path=OUTPUT_DIR + "test_dict.png",
    )

    # images as str
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        draw=True,
        path=OUTPUT_DIR + "test_str.png",
    )

    # images as path
    clustree(
        data=iris_data,
        prefix="K",
        images=Path(INPUT_DIR),
        draw=True,
        path=OUTPUT_DIR + "test_path.png",
    )
