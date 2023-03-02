import matplotlib.pyplot as plt

INPUT_DIR = "tests/data/input/"

OUTPUT_DIR = "tests/data/output/"


def add_title_to_fig(path: str, title: str) -> None:
    to_edit = plt.imread(path + ".png")
    fig, ax = plt.subplots()
    ax.imshow(to_edit)
    fig.suptitle(title, fontsize=6)
    ax.axis("off")
    plt.savefig(path, dpi=200, bbox_inches="tight")
