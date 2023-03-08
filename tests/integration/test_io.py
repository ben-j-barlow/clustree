import os

from clustree._graph import clustree
from tests.helpers import INPUT_DIR, OUTPUT_DIR, add_title_to_fig


def test_read_images_from_0(iris_data_0):
    min_k_lower = 0

    assert not os.path.isfile(INPUT_DIR + "1_0.png")
    assert min_k_lower == 0

    title = "Node images generated for k = 0,...,(K-1)"
    output = OUTPUT_DIR + "io_generate_imgs_from_0"

    clustree(
        data=iris_data_0,
        prefix="K",
        images=INPUT_DIR,
        output_path=output,
        min_cluster_number=min_k_lower,
        errors=False,
    )
    add_title_to_fig(path=output, title=title)
