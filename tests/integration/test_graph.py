from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from clustree._graph import clustree
from clustree._hash import hash_node_id
from tests.helpers import INPUT_DIR, OUTPUT_DIR, add_title_to_fig

IMG_FILES = pytest.mark.datafiles(
    INPUT_DIR + "1_1.png",
    INPUT_DIR + "2_1.png",
    INPUT_DIR + "2_2.png",
    INPUT_DIR + "3_1.png",
    INPUT_DIR + "3_2.png",
    INPUT_DIR + "3_3.png",
)


@IMG_FILES
def test_clustree_draw_dict(datafiles, iris_data):
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


def test_clustree_draw_str(datafiles, iris_data):
    # images as str
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        draw=True,
        output_path=OUTPUT_DIR + "test_str.png",
    )


def test_clustree_draw_path(datafiles, iris_data):
    # images as path
    clustree(
        data=iris_data,
        prefix="K",
        images=Path(INPUT_DIR),
        draw=True,
        output_path=OUTPUT_DIR + "test_path.png",
    )


def test_orientation_horizontal(iris_data):
    output = OUTPUT_DIR + "test_horizontal"
    title = "Horizontal plot (left to right)"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        output_path=output,
        orientation="horizontal",
    )
    add_title_to_fig(path=output, title=title)
