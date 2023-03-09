import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from networkx import DiGraph, draw_networkx_edges, get_edge_attributes

from clustree._clustree_typing import ORIENTATION_INPUT_TYPE, OUTPUT_PATH_TYPE


def ig_node_name_to_id(name, g):
    return g.vs.find(name=name).index


def get_pos(dg: DiGraph, orientation: ORIENTATION_INPUT_TYPE) -> dict[int, np.ndarray]:
    """
    Produce position of each node. Use multipartite_layout, scale resulting positions \
    to [0, 1] interval and flip graph vertically by returning (1 - pos). This forces \
    the tree to position the root node at the top rather than the bottom.

    :return: (x, y) coordinates of nodes
    Parameters
    ----------
    dg
        Clustree.

    Returns
    -------
    dict[int, np.ndarray]
        Dictionary of form node_id: (x, y). x,y in [0, 1].

    """

    g = ig.Graph()
    nodes = list(dg.nodes)
    g.add_vertices(n=nodes)
    edges = list(dg.edges)
    ls_as_id = [
        (ig_node_name_to_id(name=node_from, g=g), ig_node_name_to_id(name=node_to, g=g))
        for node_from, node_to in edges
    ]
    g.add_edges(ls_as_id)
    layout = g.layout_reingold_tilford(root=[0])
    pos = {k: v for k, v in zip(nodes, layout.coords)}

    x_vals, y_vals = [v[0] for k, v in pos.items()], [v[1] for k, v in pos.items()]
    min_y, max_y = min(y_vals), max(y_vals)
    min_x, max_x = min(x_vals), max(x_vals)

    norm_x = [(x - min_x) / (max_x - min_x) for x in x_vals]
    norm_y = [(y - min_y) / (max_y - min_y) for y in y_vals]
    if orientation == "vertical":
        return {k: (x, 1 - y) for k, x, y in zip(list(pos.keys()), norm_x, norm_y)}
    return {k: (y, x) for k, x, y in zip(list(pos.keys()), norm_x, norm_y)}


def draw_clustree(
    dg: DiGraph,
    path: OUTPUT_PATH_TYPE,
    orientation: ORIENTATION_INPUT_TYPE,
):
    pos = get_pos(dg=dg, orientation=orientation)
    draw_with_images(dg=dg, pos=pos)
    if path:
        plt.savefig(path, dpi=400, bbox_inches="tight")


def draw_with_images(
    dg: DiGraph,
    pos: dict[int, np.ndarray],
    icon_size: float = 0.04,
):
    fig, ax = plt.subplots()

    node_shape = "s"
    colors = get_edge_attributes(dg, "edge_color").values()
    alpha = get_edge_attributes(dg, "alpha").values()

    draw_networkx_edges(
        G=dg, pos=pos, node_shape=node_shape, edge_color=colors, alpha=list(alpha)
    )

    tr_figure = ax.transData.transform
    tr_axes = fig.transFigure.inverted().transform

    ax_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    icon_size *= ax_range
    icon_center = icon_size / 2.0

    for n in dg.nodes:
        xf, yf = tr_figure(pos[n])
        xa, ya = tr_axes((xf, yf))
        a = plt.axes([xa - icon_center, ya - icon_center, icon_size, icon_size])

        a.imshow(dg.nodes[n]["image_with_drawing"])

        a.patch.set_visible(False)
        a.axis("off")
    ax.axis("off")
