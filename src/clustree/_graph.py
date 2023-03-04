import matplotlib.pyplot as plt
import numpy as np
from networkx import (
    DiGraph,
    draw_networkx_edges,
    get_edge_attributes,
    multipartite_layout,
)

from clustree._clustree_typing import (
    CMAP_TYPE,
    COLOR_AGG_TYPE,
    DATA_INPUT_TYPE,
    EDGE_COLOR_TYPE,
    IMAGE_INPUT_TYPE,
    NODE_COLOR_TYPE,
    OUTPUT_PATH_TYPE,
)
from clustree._config import ClustreeConfig
from clustree._handle_pars import get_and_check_cluster_cols, handle_data, handle_images


def clustree(
    data: DATA_INPUT_TYPE,
    prefix: str,
    images: IMAGE_INPUT_TYPE,
    output_path: OUTPUT_PATH_TYPE = None,
    draw: bool = True,
    node_color: NODE_COLOR_TYPE = None,  # if None, set equal to prefix
    node_color_aggr: COLOR_AGG_TYPE = None,
    node_cmap: CMAP_TYPE = None,
    edge_color: EDGE_COLOR_TYPE = "samples",
    edge_cmap: CMAP_TYPE = None,
    errors: bool = False,
    orientation: str = "vertical",
) -> DiGraph:
    """
    Create a plot of a clustering tree showing the relationship between clusterings \
    at different resolutions.

    Parameters
    ----------
    data : Union[str, Path, DataFrame]
        Path of csv or DataFrame object.
    prefix : str
        String indicating columns containing clustering information.
    images : Union[str, Path, dict[str, ndarray]
        String indicating directory containing images. See more information on \
         files expected in directory in Notes.
    output_path : Union[str, Path], optional
        Directory to output the final plot to. If None, then output not wrriten to file.
    draw : bool, optional
        Whether to draw the clustree. Defaults to True. If False and output_path \
        supplied, will be overridden. Saving to file requires drawing.
    node_color : Any, optional
        For continuous colormap, use 'samples' or the name of a metadata column to \
        color nodes by. For discrete colors, parse the same value parsed to prefix to \
        color by resolution or specify a fixed color (see Specifying colors in \
        Matplotlib tutorial here: \
        https://matplotlib.org/stable/tutorials/colors/colors.html). If None, default \
        set equal to value of prefix to color by resolution.
    node_color_aggr : Union[Callable, str], optional
        If node_color is a column name then a function or string giving the name of a \
        function to aggregate that column for samples in each cluster.
    node_cmap : Union[mpl.colors.Colormap, str], optional
        If node_color is 'samples' or a column name then a colourmap to use (see \
        Colormap Matplotlib tutorial here: \
        https://matplotlib.org/stable/tutorials/colors/colormaps.html).
    edge_color : Any, optional
        For continuous colormap, use 'samples'. For discrete colors, use 'prefix' to \
        color by resolution or specify a fixed color (see Specifying colors in \
        Matplotlib tutorial here: \
        https://matplotlib.org/stable/tutorials/colors/colors.html). If None, default \
        set to 'samples'.
    edge_cmap : Union[mpl.colors.Colormap, str], optional
        If edge_color is 'samples' then a colourmap to use (see Colormap Matplotlib \
        tutorial here: https://matplotlib.org/stable/tutorials/colors/colormaps.html).
    errors : bool, optional
        Whether to raise an error if an image is missing from directory supplied to
        images parameter. If False, a fake image will be created with text 'K_k' \
        where K is cluster resolution and k is cluster number. Defaults to False.
    orientation: str, optional
        'vertical' or 'horizontal' to control orientation in which samples flow \
        through the graph. Defaults to 'vertical'.

    Returns
    -------
    DiGraph
        The directed graph that represents the clustree.

    Notes
    -------
    The function must be supplied (a directory for) data, a prefix that indicates \
    where to find cluster membership, and (a directory for) images.

    The function that reads images from given directory will determine maximum cluster \
    resolution kk from the data and look for files named 'K_k' with 1 <= K <= kk and, \
    for each K, 1 <= k <= K. There are requirements on the images in the directory:

        - Their file name must be K_k.png.
        - K is cluster resolution and k is cluster number.
        - Only .png files are accepted.
        - If a file or more is missing, behaviour will depend on errors parameter.
        - Other files in the directory will be ignored.

    """
    _data = handle_data(data=data)
    kk = get_and_check_cluster_cols(cols=_data.columns, prefix=prefix)
    _images = handle_images(images=images, kk=kk, errors=errors)
    config = ClustreeConfig(
        image_cf=_images,
        prefix=prefix,
        kk=kk,
        data=_data,
        node_color=node_color,
        node_color_aggr=node_color_aggr,
        node_cmap=node_cmap,
        edge_color=edge_color,
        edge_cmap=edge_cmap,
    )

    dg = construct_clustree(cf=config)
    if draw or output_path:
        draw_clustree(dg=dg, path=output_path, orientation=orientation)
    return dg


def construct_clustree(cf: ClustreeConfig) -> DiGraph:
    dg = DiGraph()
    dg.add_nodes_from([(k, v) for k, v in cf.node_cf.items()])
    dg.add_edges_from([(v["start"], v["end"], v) for v in cf.edge_cf.values()])
    return dg


def get_pos(dg: DiGraph, orientation: str) -> dict[int, np.ndarray]:
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
    pos = multipartite_layout(dg, subset_key="res", align="horizontal")
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
    orientation: str,
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
