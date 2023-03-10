from typing import Optional

from networkx import DiGraph

from clustree._clustree_typing import (
    CIRCLE_POS_TYPE,
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
from clustree._handle_pars import get_and_check_cluster_cols, handle_data, handle_images


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
    errors: bool = False,
    orientation: ORIENTATION_INPUT_TYPE = "vertical",
    min_cluster_number: MIN_CLUSTER_NUMBER_TYPE = 1,
    pos: CIRCLE_POS_TYPE = "tr",
    kk: Optional[int] = None,
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
        color nodes by. For discrete colors, coloring by cluster resolution can be \
        achieved by parsing 'prefix' or the same value parsed to the prefix parameter. \
        Finally, it is possible to specify a fixed color (see Specifying colors in \
        Matplotlib tutorial here: \
        https://matplotlib.org/stable/tutorials/colors/colors.html). If None, default \
        set to 'prefix' to color by resolution.
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
    orientation : Literal['vertical', 'horizontal'], optional
        'vertical' or 'horizontal' to control orientation in which samples flow \
        through the graph. Defaults to 'vertical'.
    min_cluster_number : Literal[0, 1], optional
        Indicates if cluster numbers are 0,...,(K-1) or 1,...,K.

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
    start_at_1 = bool(min_cluster_number)

    _data = handle_data(data=data)
    kk = get_and_check_cluster_cols(cols=_data.columns, prefix=prefix, user_kk=kk)
    _images = handle_images(images=images, kk=kk, errors=errors, start_at_1=start_at_1)
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
        start_at_1=start_at_1,
    )

    dg = construct_clustree(cf=config)
    if draw or output_path:
        draw_clustree(
            dg=dg, path=output_path, orientation=orientation,
            #img_len=1 / (2 * kk), circle_pos=pos
        )
    return dg


def construct_clustree(cf: ClustreeConfig) -> DiGraph:
    dg = DiGraph()
    dg.add_nodes_from([(k, v) for k, v in cf.node_cf.items()])
    dg.add_edges_from([(v["start"], v["end"], v) for v in cf.edge_cf.values()])
    return dg
