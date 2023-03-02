import numpy as np

from clustree._graph import clustree
from tests.helpers import INPUT_DIR, OUTPUT_DIR, add_title_to_fig


def test_set_node_color_prefix(iris_data):
    title = "Node colour by res"
    output = OUTPUT_DIR + "cfg_node_col_prefix"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        node_color="K",
        output_path=output,
        edge_color="tab:blue",
    )
    add_title_to_fig(path=output, title=title)


def test_set_node_color_samples(iris_data):
    title = "Node colour by samples, colorbar to add"
    output = OUTPUT_DIR + "cfg_node_col_samples"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        node_color="samples",
        node_cmap="Oranges",
        edge_color="tab:blue",
        output_path=output,
    )
    add_title_to_fig(path=output, title=title)


def test_set_node_color_agg(iris_data):
    title = "Node colour by sepal_legnth summed, colorbar to add"
    output = OUTPUT_DIR + "cfg_node_col_agg"
    clustree(
        prefix="K",
        data=iris_data,
        images=INPUT_DIR,
        node_color="sepal_length",
        node_color_aggr=np.mean,
        node_cmap="Oranges",
        edge_color="tab:blue",
        output_path=output,
    )
    add_title_to_fig(path=output, title=title)


def test_set_node_color_fixed(iris_data):
    title = "Node colour fixed"
    output = OUTPUT_DIR + "cfg_node_col_fixed"
    clustree(
        prefix="K",
        data=iris_data,
        images=INPUT_DIR,
        node_color="C1",
        edge_color="tab:blue",
        output_path=output,
    )
    add_title_to_fig(path=output, title=title)


def test_set_edge_color_prefix(iris_data):
    title = "Edge colour by res"
    output = OUTPUT_DIR + "cfg_edge_col_prefix"
    clustree(
        prefix="K",
        data=iris_data,
        images=INPUT_DIR,
        node_color="tab:blue",
        edge_color="K",
        output_path=output,
    )
    add_title_to_fig(path=output, title=title)


def test_set_edge_color_samples(iris_data):
    title = "Edge colour by samples"
    output = OUTPUT_DIR + "cfg_edge_col_samples"
    clustree(
        prefix="K",
        data=iris_data,
        images=INPUT_DIR,
        node_color="tab:blue",
        edge_color="samples",
        output_path=output,
    )
    add_title_to_fig(path=output, title=title)
