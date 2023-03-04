import os
import tempfile
from pathlib import Path

from clustree import clustree
from tests.helpers import INPUT_DIR


def test_path_override_draw(iris_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define the path for the output file
        draw = False
        output_file = Path(temp_dir) / "test_plot.png"
        assert not os.path.isfile(output_file)
        dg = clustree(
            data=iris_data,
            prefix="K",
            images=INPUT_DIR,
            draw=draw,
            output_path=output_file,
        )
        assert os.path.isfile(output_file)
