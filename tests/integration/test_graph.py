from pathlib import Path

from clustree._graph import clustree
from tests.helpers import INPUT_DIR, OUTPUT_DIR, add_title_to_fig


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
