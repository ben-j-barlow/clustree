import typing
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from networkx import (
    DiGraph,
    draw_networkx_edges,
    get_edge_attributes,
    multipartite_layout,
)

from clustree._handle_pars import (
    append_k_k_cols,
    get_and_check_cluster_cols,
    handle_images,
)
from clustree.clustree_typing import (
    CMAP_TYPE,
    COLOR_AGG_TYPE,
    EDGE_COLOR_TYPE,
    IMAGE_INPUT_TYPE,
    NODE_COLOR_TYPE,
)
from clustree.config import ClustreeConfig

if typing.TYPE_CHECKING:
    from pandas import DataFrame


def clustree(
    data: "DataFrame",
    prefix: str,
    images: IMAGE_INPUT_TYPE,
    errors: bool = False,
    draw: bool = True,
    path: Union[
        str, None
    ] = "/Users/benbarlow/dev/clustree/tests/data/output/mytest.png",
    node_color: NODE_COLOR_TYPE = None,
    node_color_aggr: COLOR_AGG_TYPE = None,
    node_cmap: CMAP_TYPE = None,
    edge_color: EDGE_COLOR_TYPE = None,
    edge_cmap: CMAP_TYPE = None,
) -> DiGraph:
    cols, kk = get_and_check_cluster_cols(cols=data.columns, prefix=prefix)
    data = append_k_k_cols(data=data, prefix=prefix, kk=kk)
    _images = handle_images(images=images, kk=kk, errors=errors)
    config = ClustreeConfig(
        image_cf=_images,
        prefix=prefix,
        kk=kk,
        data=data,
        node_color=node_color,
        node_color_aggr=node_color_aggr,
        node_cmap=node_cmap,
        edge_color=edge_color,
        edge_cmap=edge_cmap,
    )

    dg = construct_clustree(cf=config)
    if draw:
        draw_clustree(dg=dg, path=path)
    return dg


def construct_clustree(cf: ClustreeConfig) -> DiGraph:
    dg = DiGraph()
    dg.add_nodes_from([(k, v) for k, v in cf.node_cf.items()])
    dg.add_edges_from([(v["start"], v["end"], v) for v in cf.edge_cf.values()])
    return dg


def get_pos(dg: DiGraph) -> dict[str, np.ndarray]:
    """
    Produce position of each node. Use multipartite_layout, scale resulting positions \
    to [0, 1] interval and flip graph vertically by returning (1 - pos). This forces \
    the tree to position the root node at the top rather than the bottom.

    :param dg: directed graph
    :return: (x, y) coordinates of nodes
    """
    pos = multipartite_layout(dg, subset_key="res", align="horizontal")
    y_vals = {v[1] for k, v in pos.items()}
    min_y, max_y = min(y_vals), max(y_vals)
    return {k: 1 - ((v - min_y) / (max_y - min_y)) for k, v in pos.items()}


def draw_clustree(
    dg: DiGraph,
    path: Union[
        str, None
    ] = "/Users/benbarlow/dev/clustree/tests/data/output/mytest.png",
):
    pos = get_pos(dg=dg)
    draw_with_images(dg=dg, pos=pos)
    if path:
        plt.savefig(path, dpi=400, bbox_inches="tight")


def draw_with_images(
    dg: DiGraph,
    pos: dict[str, np.ndarray],
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
