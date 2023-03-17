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


def test_node_size(iris_data):
    output = OUTPUT_DIR + "test_node_size"
    title = "Node size small"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        output_path=output,
        node_size=100,
        node_size_edge=300,
    )
    add_title_to_fig(path=output, title=title)


def test_no_border(iris_data):
    output = OUTPUT_DIR + "test_no_border"
    title = "Nodes should have no border"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        output_path=output,
        node_size=100,
        node_size_edge=300,
        border_size=0,
    )
    add_title_to_fig(path=output, title=title)

def test_no_arrows(iris_data):
    output = OUTPUT_DIR + "test_no_arrows"
    title = "Edges should have no arrows"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        output_path=output,
        node_size=100,
        node_size_edge=300,
        arrows=False,
    )
    add_title_to_fig(path=output, title=title)
