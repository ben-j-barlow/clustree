import igraph as ig
import matplotlib.pyplot as plt
from networkx import DiGraph, draw_networkx_edges, get_edge_attributes

from clustree._clustree_typing import (
    CIRCLE_POS_TYPE,
    ORIENTATION_INPUT_TYPE,
    OUTPUT_PATH_TYPE,
)


def ig_node_name_to_id(name, g):
    return g.vs.find(name=name).index


def get_pos(
    dg: DiGraph, orientation: ORIENTATION_INPUT_TYPE
) -> dict[int, tuple[float, float]]:
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
    kk: int,
    circle_pos: CIRCLE_POS_TYPE,
):
    pos = get_pos(dg=dg, orientation=orientation)
    draw_with_images(dg=dg, pos=pos, kk=kk, circle_pos=circle_pos)
    if path:
        plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()


def draw_with_images(
    dg: DiGraph,
    pos: dict[int, tuple[float, float]],
    kk: int,
    circle_pos: CIRCLE_POS_TYPE,
):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

    node_shape = "s"
    colors = get_edge_attributes(dg, "edge_color").values()
    alpha = get_edge_attributes(dg, "alpha").values()

    draw_networkx_edges(
        G=dg,
        pos=pos,
        node_shape=node_shape,
        node_size=600,
        edge_color=colors,
        alpha=list(alpha),
        ax=ax,
    )

    draw_custom_nodes(dg=dg, pos=pos, ax=ax, kk=kk, circle_pos=circle_pos)
    ax.autoscale()
    ax.axis("off")


def get_extent(bl_anchor, length):
    return [bl_anchor[0], bl_anchor[0] + length, bl_anchor[1], bl_anchor[1] + length]


def get_circle_centre(
    radius: float,
    pos: CIRCLE_POS_TYPE,
    bl_anchor: tuple[float, float],
    length: float,
) -> tuple[float, float]:
    l, b = bl_anchor
    r, t = l + length, b + length
    pos_dict = {
        "tl": (l + radius, t - radius),
        "t": (l + (length / 2), t - radius),
        "tr": (r - radius, t - radius),
        "l": (l + radius, b + (length / 2)),
        "r": (r - radius, b + (length / 2)),
        "bl": (l + radius, b + radius),
        "b": (l + (length / 2), b + radius),
        "br": (r - radius, b + radius),
    }
    to_return = pos_dict[pos]
    return to_return


def draw_custom_nodes(
    dg: DiGraph,
    pos,
    kk: int,
    ax,
    circle_pos: CIRCLE_POS_TYPE,
    circle_prop: float = 0.1,
):
    img_len = 1 / (2 * kk)
    radius = img_len * circle_prop / 2
    for node_id, attr in dg.nodes.items():
        img_cent_anchor = pos[node_id]
        img_bl_anchor = (
            img_cent_anchor[0] - (img_len / 2),
            img_cent_anchor[1] - (img_len / 2),
        )
        ax.imshow(
            attr["image"], extent=get_extent(bl_anchor=img_bl_anchor, length=img_len)
        )
        circ_cent_anchor = get_circle_centre(
            radius=radius, pos=circle_pos, bl_anchor=img_bl_anchor, length=img_len
        )
        circle = plt.Circle(
            circ_cent_anchor,
            radius=radius,
            fill=True,
            color=attr["node_color"],
        )
        ax.add_artist(circle)
