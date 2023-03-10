from clustree._graph import clustree
from tests.helpers import INPUT_DIR, OUTPUT_DIR, add_title_to_fig


def test_clustree_param_pos_tr(datafiles, iris_data):
    pos = "tr"
    title = f"Node in {pos}"
    output = OUTPUT_DIR + f"draw/test_pos_{pos}"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        pos=pos,
        output_path=output,
    )
    add_title_to_fig(title=title, path=output)


def test_clustree_param_pos_t(datafiles, iris_data):
    pos = "t"
    title = f"Node in {pos}"
    output = OUTPUT_DIR + f"draw/test_pos_{pos}"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        output_path=output,
        pos=pos,
    )
    add_title_to_fig(title=title, path=output)


def test_clustree_param_pos_tl(datafiles, iris_data):
    pos = "tl"
    title = f"Node in {pos}"
    output = OUTPUT_DIR + f"draw/test_pos_{pos}"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        output_path=output,
        pos=pos,
    )
    add_title_to_fig(title=title, path=output)


def test_clustree_param_pos_l(datafiles, iris_data):
    pos = "l"
    title = f"Node in {pos}"
    output = OUTPUT_DIR + f"draw/test_pos_{pos}"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        output_path=output,
        pos=pos,
    )
    add_title_to_fig(title=title, path=output)


def test_clustree_param_pos_r(datafiles, iris_data):
    pos = "r"
    title = f"Node in {pos}"
    output = OUTPUT_DIR + f"draw/test_pos_{pos}"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        output_path=output,
        pos=pos,
    )
    add_title_to_fig(title=title, path=output)


def test_clustree_param_pos_bl(datafiles, iris_data):
    pos = "bl"
    title = f"Node in {pos}"
    output = OUTPUT_DIR + f"draw/test_pos_{pos}"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        output_path=output,
        pos=pos,
    )
    add_title_to_fig(title=title, path=output)


def test_clustree_param_pos_b(datafiles, iris_data):
    pos = "b"
    title = f"Node in {pos}"
    output = OUTPUT_DIR + f"draw/test_pos_{pos}"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        output_path=output,
        pos=pos,
    )
    add_title_to_fig(title=title, path=output)


def test_clustree_param_pos_br(datafiles, iris_data):
    pos = "br"
    title = f"Node in {pos}"
    output = OUTPUT_DIR + f"draw/test_pos_{pos}"
    clustree(
        data=iris_data,
        prefix="K",
        images=INPUT_DIR,
        output_path=output,
        pos=pos,
    )
    add_title_to_fig(title=title, path=output)
