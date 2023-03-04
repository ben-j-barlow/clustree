from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from clustree._graph import clustree
from clustree._hash import hash_node_id
from tests.helpers import INPUT_DIR, OUTPUT_DIR

IMG_FILES = pytest.mark.datafiles(
    INPUT_DIR + "1_1.png",
    INPUT_DIR + "2_1.png",
    INPUT_DIR + "2_2.png",
    INPUT_DIR + "3_1.png",
    INPUT_DIR + "3_2.png",
    INPUT_DIR + "3_3.png",
)


@IMG_FILES
def test_clustree_draw(datafiles, iris_data):
    to_read = [Path(ele) for ele in datafiles.listdir()]
    img_files = {
        hash_node_id(int(file.stem[0]), int(file.stem[2])): plt.imread(file)
        for file in to_read
    }

    # images as dict
    clustree(
        data=iris_data,
        prefix="K",
        images=img_files,
        draw=True,
        output_path=OUTPUT_DIR + "test_dict.png",
    )

    # images as str
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        draw=True,
        output_path=OUTPUT_DIR + "test_str.png",
    )

    # images as path
    clustree(
        data=iris_data,
        prefix="K",
        images=Path(INPUT_DIR),
        draw=True,
        output_path=OUTPUT_DIR + "test_path.png",
    )
