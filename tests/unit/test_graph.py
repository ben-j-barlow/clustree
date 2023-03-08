import os
import tempfile
from pathlib import Path

import pytest

from clustree._graph import clustree
from clustree._hash import hash_node_id
from tests.helpers import INPUT_DIR


def test_clustree(iris_data):
    dg = clustree(
        data=iris_data, prefix="K", images=INPUT_DIR, draw=False, output_path=None
    )

    assert dg.number_of_edges() == 6
    assert dg.number_of_nodes() == 6
    assert set(dg.edges) == {
        (hash_node_id(1, 1), hash_node_id(2, 1)),
        (hash_node_id(1, 1), hash_node_id(2, 2)),
        (hash_node_id(2, 1), hash_node_id(3, 1)),
        (hash_node_id(2, 1), hash_node_id(3, 2)),
        (hash_node_id(2, 2), hash_node_id(3, 2)),
        (hash_node_id(2, 2), hash_node_id(3, 3)),
    }


def test_clustree_start_at_0(iris_data_0):
    min_k_lower = 0
    assert min_k_lower == 0
    dg = clustree(
        data=iris_data_0,
        prefix="K",
        images=INPUT_DIR,
        draw=False,
        output_path=None,
        min_cluster_number=min_k_lower,
        errors=False,
    )

    assert dg.number_of_edges() == 6
    assert dg.number_of_nodes() == 6
    assert set(dg.edges) == {
        (hash_node_id(1, 0), hash_node_id(2, 0)),
        (hash_node_id(1, 0), hash_node_id(2, 1)),
        (hash_node_id(2, 0), hash_node_id(3, 0)),
        (hash_node_id(2, 0), hash_node_id(3, 1)),
        (hash_node_id(2, 1), hash_node_id(3, 1)),
        (hash_node_id(2, 1), hash_node_id(3, 2)),
    }


def test_clustree_start_at_0_err(iris_data_0):
    min_k_lower = 0

    assert not os.path.isfile(INPUT_DIR + "1_0.png")
    assert min_k_lower == 0

    with pytest.raises(FileNotFoundError):
        clustree(
            data=iris_data_0,
            prefix="K",
            images=INPUT_DIR,
            draw=False,
            output_path=None,
            min_cluster_number=min_k_lower,
            errors=True,
        )


def test_path_override_draw(iris_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define the path for the output file
        draw = False
        output_file = Path(temp_dir) / "test_plot.png"
        assert not os.path.isfile(output_file)
        clustree(
            data=iris_data,
            prefix="K",
            images=INPUT_DIR,
            draw=draw,
            output_path=output_file,
        )
        assert os.path.isfile(output_file)
