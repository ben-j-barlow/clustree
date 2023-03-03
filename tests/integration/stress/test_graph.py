from clustree._graph import clustree
from tests.helpers import INPUT_DIR, OUTPUT_DIR


def test_many_nodes_time(iris_data):
    clustree(
        data=iris_data,
        prefix="k",
        images=INPUT_DIR,
        draw=True,
        output_path=OUTPUT_DIR + "many_nodes.png",
        errors=False,
    )
