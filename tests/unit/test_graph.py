import os
import tempfile
from pathlib import Path

from clustree import clustree, hash_node_id
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