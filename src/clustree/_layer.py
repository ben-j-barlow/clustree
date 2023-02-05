import tempfile
from typing import List, Set, Union

import matplotlib.pyplot as plt
import numpy as np
from networkx import DiGraph

from clustree._handle_pars import to_k_k


def add_layer(
    k_upper: int,
    prev_cluster: Union[List[str], None],
    cluster: List[str],
    images: dict[str, np.ndarray],
    dg: DiGraph,
) -> None:
    """
    Add a new cluster resolution to the clustree.

    :param k_upper: cluster resolution
    :param prev_cluster: cluster membership at previous resolution
    :param cluster: cluster membership at this resolution
    :param dg: directed graph
    :return: None
    """
    if k_upper == 1:
        dg.add_node("1_1")
    else:
        add_edges(prev_cluster=prev_cluster, cluster=cluster, dg=dg)
    set_res(k_upper=k_upper, to_set=set(cluster), dg=dg)
    if images:
        set_img(images=images, dg=dg, k_upper=k_upper)


def add_edges(prev_cluster: List[str], cluster: List[str], dg: DiGraph) -> None:
    """
    Add edges from previous resolution to current resolution.

    :param prev_cluster: cluster membership at previous resolution
    :param cluster: cluster membership at this resolution
    :param dg: directed graph
    :return: None
    """
    edges_to_add = list(set([(pc, c) for pc, c in zip(prev_cluster, cluster)]))
    dg.add_edges_from(edges_to_add)


def set_res(k_upper: int, to_set: Set[str], dg: DiGraph) -> None:
    """
    Set custom node attribute 'res' for graph drawing.

    :param k_upper: cluster resolution
    :param to_set: nodes to set attribute of, e.g., ['2_1', '2_2']
    :param dg: directed graph
    :return: None
    """
    for node in to_set:
        dg.nodes[node]["res"] = k_upper


def set_img(
    images: dict[str, np.ndarray], dg: DiGraph, k_upper: int, radius: float = 0.1
) -> None:
    for k_lower, img in images.items():
        dg.nodes[to_k_k(k_upper=k_upper, k_lower=k_lower)]["image"] = draw_circle(
            img=img, radius=radius
        )


def draw_circle(
    img: np.ndarray,
    radius: float = 0.1,
    node_color: str = "red",
    img_width: int = 100,
    img_height: int = 100,
) -> np.ndarray:
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, img_width, 0, img_height])

    # Calculate radius and coordinates
    radius *= img_width
    x0 = img_width - radius
    y0 = img_height - radius

    # Add a circle at the top right of the image
    circle = plt.Circle((x0, y0), radius=radius, fill=True, color=node_color)
    ax.add_artist(circle)

    # save and read in
    fig.patch.set_visible(False)
    ax.axis("off")
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        file_path = f.name
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        to_return = plt.imread(file_path)
        plt.close()
        return to_return
