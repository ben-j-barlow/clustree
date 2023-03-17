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
    node_cmap: CMAP_TYPE = "inferno",
    edge_color: EDGE_COLOR_TYPE = "samples",
    edge_cmap: CMAP_TYPE = "viridis",
    orientation: ORIENTATION_INPUT_TYPE = "vertical",
    layout_reingold_tilford: bool = None,
    min_cluster_number: MIN_CLUSTER_NUMBER_TYPE = 1,
    border_size: float = 0.05,
    figsize: tuple[float, float] = None,
    arrows: bool = None,
    node_size: float = 300,
    node_size_edge: Optional[float] = None,
    dpi: float = 500,
    kk: Optional[int] = None,
) -> DiGraph:
    """

    Parameters
    ----------
    data : Union[Path, str]
        Path of csv or DataFrame object.
    prefix : str
        String indicating columns containing clustering information.
    images : Union[Path, str]
        Path of directory that contains images.
    output_path : Union[Path, str], optional
        Absolute path to save clustree drawing at. If file extension is supplied, must \
        be .png. If None, then output not written to file.
    draw : bool
        Whether to draw the clustree. Defaults to True. If False and output_path \
        supplied, will be overridden.
    node_color : str
        For continuous colormap, use 'samples' or the name of a metadata column to \
        color nodes by. For discrete colors, use 'prefix' to color by resolution or \
        specify a fixed color (see Specifying colors in Matplotlib tutorial here: \
        https://matplotlib.org/stable/tutorials/colors/colors.html). If None, default \
        set equal to value of prefix to color by resolution.
    node_color_aggr : Union[Callable, str], optional
        If node_color is a column name then a function or string giving the name of a \
        function to aggregate that column for samples in each cluster.
    node_cmap : Union[mpl.colors.Colormap, str]
        If node_color is 'samples' or a column name then a colourmap to use (see \
        Colormap Matplotlib tutorial here: \
        https://matplotlib.org/stable/tutorials/colors/colormaps.html).
    edge_color : str
        For continuous colormap, use 'samples'. For discrete colors, use 'prefix' to \
        color by resolution or specify a fixed color (see Specifying colors in \
        Matplotlib tutorial here: \
        https://matplotlib.org/stable/tutorials/colors/colors.html). If None, default \
        set to 'samples'.
    edge_cmap : Union[mpl.colors.Colormap, str]
        If edge_color is 'samples' then a colourmap to use (see Colormap Matplotlib \
        tutorial here: https://matplotlib.org/stable/tutorials/colors/colormaps.html).
    orientation : Literal["vertical", "horizontal"]
        Orientation of clustree drawing. Defaults to 'vertical'.
    layout_reingold_tilford : bool, optional
        Whether to use the Reingold-Tilford algorithm for node positioning. Defaults \
        to True if (kk <= 12), False otherwise. Setting True not recommended if \
        (kk > 12) due to memory bottleneck in igraph dependency.
    min_cluster_number : Literal[0, 1]
        0 if cluster number is (0, ..., K-1) or 1 if (1, ..., K). Defaults to 1.
    border_size : float
        Border width as proportion of image width. Defaults to 0.05.
    figsize : tuple[float, float]
        Parsed to matplotlib to determine figure size. Defaults to (kk/2, kk/2), \
        clipped to a minimum of (3,3) and maximum of (10,10).
    arrows : bool
        Whether to add arrows to graph edges. Removing arrows alleviates appearance \
        issue caused by arrows overlapping nodes. Defaults to True.
    node_size : float
        Size of nodes in clustree graph drawing. Parsed directly to \
        networkx.draw_networkx_nodes. Deafult to 300.
    node_size_edge: float
        Controls edge start and end point. Parsed directly to \
        networkx.draw_networkx_edges.
    dpi : float
        Controls resolution of output if saved to file.
    kk : int, optional
        Choose custom depth of clustree graph.

    Returns
    -------
    networkx.DiGraph
        Clustree drawing.

    Notes
    -------
    Given data (containing cluster membership for K in 1, ..., kk) and a set of images,\
 plot a graph showing the clustering tree. Edges show samples moving from a cluster at \
 one resolution to a cluster at another resolution. Nodes are displayed as images \
 chosen by user.

    The directory of images should contain files in format 'K_k.png', where K \
    (in 1, ..., kk) is cluster resolution and k (in 1, ..., K) is cluster number. \
    Alternatively, by setting the parameter min_cluster_number = 0, k is expected to \
    take values in 0, ..., K-1.
    """

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
        node_size_edge = 3 * node_size

    if layout_reingold_tilford is None:
        layout_reingold_tilford = False
        if kk < 13:
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
