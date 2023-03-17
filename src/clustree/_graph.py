from typing import Optional

from networkx import DiGraph

from clustree._clustree_typing import (
    CMAP_TYPE,
    COLOR_AGG_TYPE,
    DATA_INPUT_TYPE,
    EDGE_COLOR_TYPE,
    IMAGE_INPUT_TYPE,
    MIN_CLUSTER_NUMBER_TYPE,
    NODE_COLOR_TYPE,
    ORIENTATION_INPUT_TYPE,
    OUTPUT_PATH_TYPE,
)
from clustree._config import ClustreeConfig
from clustree._draw import draw_clustree
from clustree._handle_pars import get_and_check_cluster_cols, handle_data


def clustree(
    data: DATA_INPUT_TYPE,
    prefix: str,
    images: IMAGE_INPUT_TYPE,
    output_path: OUTPUT_PATH_TYPE = None,
    draw: bool = True,
    node_color: NODE_COLOR_TYPE = "prefix",
    node_color_aggr: COLOR_AGG_TYPE = None,
    node_cmap: CMAP_TYPE = None,
    edge_color: EDGE_COLOR_TYPE = "samples",
    edge_cmap: CMAP_TYPE = None,
    orientation: ORIENTATION_INPUT_TYPE = "vertical",
    layout_reingold_tilford: bool = None,
    min_cluster_number: MIN_CLUSTER_NUMBER_TYPE = 1,
    border_size: float = 0.05,
    figsize=None,
    arrows=None,
    node_size: float = 300,
    node_size_edge: float = None,
    dpi=500,
    kk: Optional[int] = None,
) -> DiGraph:

    _data = handle_data(data=data)
    kk = get_and_check_cluster_cols(cols=_data.columns, prefix=prefix, user_kk=kk)

    border_size = float(border_size)
    images = str(images)
    if images[-1] != "/":
        images = images + "/"
    start_at_1 = bool(min_cluster_number)
    if not figsize:
        if kk < 6:
            figsize = (3, 3)
        else:
            w = min([kk, 20]) / 2
            figsize = (w, w)
    if arrows is None:
        arrows = False
        if kk <= 10:
            arrows = True
    if not node_size_edge:
        node_size_edge = 2 * node_size

    if not layout_reingold_tilford:
        layout_reingold_tilford = False
        if kk < 10:
            layout_reingold_tilford = True

    config = ClustreeConfig(
        prefix=prefix,
        kk=kk,
        data=_data,
        node_color=node_color,
        node_color_aggr=node_color_aggr,
        node_cmap=node_cmap,
        edge_color=edge_color,
        edge_cmap=edge_cmap,
        start_at_1=start_at_1,
    )

    dg = construct_clustree(cf=config)
    if draw or output_path:
        draw_clustree(
            dg=dg,
            path=output_path,
            orientation=orientation,
            rt_layout=layout_reingold_tilford,
            images=images,
            figsize=figsize,
            node_size=node_size,
            node_size_edge=node_size_edge,
            dpi=dpi,
            border_size=border_size,
            arrows=arrows,
            node_color_sm=config.node_color_sm,
            edge_color_sm=config.edge_color_sm,
            node_color_title=config.node_color_legend_title,
            edge_color_title=config.edge_color_legend_title,
        )
    return dg


def construct_clustree(cf: ClustreeConfig) -> DiGraph:
    dg = DiGraph()
    dg.add_nodes_from([(k, v) for k, v in cf.node_cf.items()])
    dg.add_edges_from([(v["start"], v["end"], v) for v in cf.edge_cf.values()])
    return dg
