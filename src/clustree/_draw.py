from typing import Sequence, Union

import cv2
import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.path import get_path_collection_extents
from networkx import DiGraph, draw_networkx_edges, get_edge_attributes

from clustree._clustree_typing import (
    IMAGE_INPUT_TYPE,
    ORIENTATION_INPUT_TYPE,
    OUTPUT_PATH_TYPE,
)


def ig_node_name_to_id(name, g):
    return g.vs.find(name=name).index


def get_pos(
    dg: DiGraph, orientation: ORIENTATION_INPUT_TYPE, rt_layout: bool
) -> dict[int, tuple[float, float]]:

    if rt_layout:
        nodes = list(dg.nodes)
        edges = list(dg.edges)

        g = ig.Graph()
        g.add_vertices(n=nodes)
        ls_as_id = [
            (
                ig_node_name_to_id(name=node_from, g=g),
                ig_node_name_to_id(name=node_to, g=g),
            )
            for node_from, node_to in edges
        ]
        g.add_edges(ls_as_id)
        layout = g.layout_reingold_tilford(root=[0])
        pos = {k: v for k, v in zip(nodes, layout.coords)}
    else:
        pos = nx.multipartite_layout(dg, "res")
    x_vals, y_vals = [v[0] for k, v in pos.items()], [v[1] for k, v in pos.items()]
    min_y, max_y = min(y_vals), max(y_vals)
    min_x, max_x = min(x_vals), max(x_vals)

    norm_x = [(x - min_x) / (max_x - min_x) for x in x_vals]
    norm_y = [(y - min_y) / (max_y - min_y) for y in y_vals]
    if orientation == "vertical":
        return {k: (x, 1 - y) for k, x, y in zip(list(pos.keys()), norm_x, norm_y)}
    return {k: (y, x) for k, x, y in zip(list(pos.keys()), norm_x, norm_y)}


def getbb(sc, ax):
    """Function to return a list of bounding boxes in data coordinates
    for a scatter plot"""
    ax.figure.canvas.draw()  # need to draw before the transforms are set.
    transform = sc.get_transform()
    transOffset = sc.get_offset_transform()
    offsets = sc._offsets
    paths = sc.get_paths()
    transforms = sc.get_transforms()

    if not transform.is_affine:
        paths = [transform.transform_path_non_affine(p) for p in paths]
        transform = transform.get_affine()
    if not transOffset.is_affine:
        offsets = transOffset.transform_non_affine(offsets)
        transOffset = transOffset.get_affine()

    if isinstance(offsets, np.ma.MaskedArray):
        offsets = offsets.filled(np.nan)

    bboxes = []

    if len(paths) and len(offsets):
        if len(paths) < len(offsets):
            # for usual scatters you have one path, but several offsets
            paths = [paths[0]] * len(offsets)
        if len(transforms) < len(offsets):
            # often you may have a single scatter size, but several offsets
            transforms = [transforms[0]] * len(offsets)

        for p, o, t in zip(paths, offsets, transforms):
            result = get_path_collection_extents(
                transform.frozen(), [p], [t], [o], transOffset.frozen()
            )
            bboxes.append(result.transformed(ax.transData.inverted()))

    return bboxes


def bb_to_extent(bb):
    bb = bb._points
    l, b = bb[0][0], bb[0][1]
    r, t = bb[1][0], bb[1][1]
    return (l, r, b, t)


def get_nodes_bbox(dg, pos, figsize, node_size, node_size_edge):
    fig, ax = plt.subplots(figsize=figsize)

    # draw_edges
    draw_networkx_edges(G=dg, pos=pos, node_shape="s", node_size=node_size_edge, ax=ax)

    # draw scatter
    nodelist = list(dg)
    xy = np.asarray([pos[v] for v in nodelist])
    node_color = "#1f78b4"
    alpha = None
    cmap = None
    vmin = None
    vmax = None
    ax = ax
    linewidths = None
    edgecolors = None
    label = None

    node_collection = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=node_size,
        c=node_color,
        marker="s",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        linewidths=linewidths,
        edgecolors=edgecolors,
        label=label,
    )

    # use get_bb
    bbox = {
        node_id: bb for node_id, bb in zip(nodelist, getbb(sc=node_collection, ax=ax))
    }
    extent = {node_id: bb_to_extent(ele) for node_id, ele in bbox.items()}
    plt.close()
    return extent


def draw_custom_nodes(
    dg: DiGraph,
    extent: Sequence[float],
    path: IMAGE_INPUT_TYPE,
    ax: plt.Axes,
    border_size_prop: float,
):
    for node_id, attr in dg.nodes.data():
        file_name: str = f"{attr['res']}_{attr['k']}.png"
        img_path = path + file_name
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if border_size_prop == float(0):
            ax.imshow(img, extent=extent[node_id], aspect=1, origin="upper", zorder=2)
        else:
            border_size = int(img.shape[0] * border_size_prop)
            border_color = tuple(val * 255 for val in attr["node_color"])
            img_with_border = cv2.copyMakeBorder(
                img,
                border_size,
                border_size,
                border_size,
                border_size,
                cv2.BORDER_CONSTANT,
                value=border_color,
            )
            ax.imshow(
                img_with_border,
                extent=extent[node_id],
                aspect=1,
                origin="upper",
                zorder=2,
            )
    ax.autoscale()


def draw_clustree(
    dg: DiGraph,
    path: OUTPUT_PATH_TYPE,
    images: IMAGE_INPUT_TYPE,
    orientation: ORIENTATION_INPUT_TYPE,
    rt_layout: bool,
    figsize: tuple[Union[int, float], Union[int, float]],
    node_size: Union[int, float],
    node_size_edge,
    arrows: bool,
    border_size: float,
    dpi: int = 500,
):

    pos = get_pos(dg=dg, orientation=orientation, rt_layout=rt_layout)
    extent = get_nodes_bbox(
        dg=dg,
        pos=pos,
        figsize=figsize,
        node_size=node_size,
        node_size_edge=node_size_edge,
    )

    fig, ax = plt.subplots()

    colors = get_edge_attributes(dg, "edge_color").values()
    alpha = get_edge_attributes(dg, "alpha").values()
    draw_networkx_edges(
        G=dg,
        pos=pos,
        node_shape="s",
        node_size=node_size_edge,
        arrows=arrows,
        ax=ax,
        edge_color=colors,
        alpha=list(alpha),
    )
    draw_custom_nodes(
        dg=dg,
        extent=extent,
        path=images,
        ax=ax,
        border_size_prop=border_size,
    )
    if path:
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
