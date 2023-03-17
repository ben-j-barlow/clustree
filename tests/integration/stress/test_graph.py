# import pandas as pd

from clustree._graph import clustree

INPUT_DIR = "tests/data/input/"

OUTPUT_DIR = "tests/data/output/"


def test_many_nodes_time(iris_data):
    clustree(
        data=iris_data,
        prefix="k",
        images=INPUT_DIR,
        draw=True,
        output_path=OUTPUT_DIR + "many_nodes.png",
        orientation="vertical",
    )


# iris_data = pd.read_csv("/Users/benbarlow/dev/clustree/tests/data/input/iris.csv")

# stress test
# clustree(
#    data=iris_data,
#    prefix="k",
#    images=INPUT_DIR,
#    kk=13,
#    draw=False,
#    layout_reingold_tilford=False,
#    orientation="vertical",
#    output_path=OUTPUT_DIR + "many_nodes.png",
# )
