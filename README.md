# clustree

## Status

**Functionality: Implemented**

* Directed graph representing clustree. Nodes are parsed images and node information is encoded by a border surrounding the image.
* Loading: Data provided directly or through a path to parent directory. Images provided through a path to parent directory.
* Appearance: Edge and node color can correspond to one of: #samples that pass through edge/node, cluster resolution `K`, or a fixed color. In the case of node color, a column name in the data and aggregate function can be used too. Use of column name and #samples creates a continuous colormap, whilst the other options result in discrete colors.
* Layout: Reingold-Tilford algorithm used for node positioning. Not recommended for kk > 12 due to memory bottleneck in igraph dependency.
* Legend: demonstration of node / edge color.


**Functionality: To Add**

* Legend: demonstration of transparency of edges.
* Layout: Bespoke implementation of Reingold-Tilford algorithm to overcome dependency's memory bottleneck.

## Usage

### Installation

Install the package with pip:

```
pip install clustree
```

### Quickstart

The powerhouse function of the library is `clustree`. Use

```
from clustree import clustree
```

to import the function. A detailed description of the parameters is provided below.

```
def clustree(
    data: Union[Path, str],
    prefix: str,
    images: Union[Path, str],
    output_path: Optional[Union[Path, str]] = None,
    draw: bool = True,
    node_color: str = "prefix",
    node_color_aggr: Optional[Union[Callable, str]] = None,
    node_cmap: Union[mpl.colors.Colormap, str] = "inferno",
    edge_color: str = "samples",
    edge_cmap: Union[mpl.colors.Colormap, str] = "viridis",
    orientation: Literal["vertical", "horizontal"] = "vertical",
    layout_reingold_tilford: bool = None,
    min_cluster_number: Literal[0, 1] = 1,
    border_size: float = 0.05,
    figsize: tuple[float, float] = None,
    arrows: bool = None,
    node_size: float = 300,
    node_size_edge: Optional[float] = None,
    dpi: float = 500,
    kk: Optional[int] = None,
) -> DiGraph:
    """

```

* `data` : Path of csv or DataFrame object.
* `prefix` : String indicating columns containing clustering information.
* `images` : Path of directory that contains images.
* `output_path` : Absolute path to save clustree drawing at. If file extension is supplied, must be .png. If None, then output not written to file.
* `draw` : Whether to draw the clustree. Defaults to True. If False and output_path supplied, will be overridden.
* `node_color` : For continuous colormap, use 'samples' or the name of a metadata column to color nodes by. For discrete colors, use 'prefix' to color by resolution or specify a fixed color (see Specifying colors in Matplotlib tutorial here: https://matplotlib.org/stable/tutorials/colors/colors.html). If None, default set equal to value of prefix to color by resolution.
* `node_color_aggr` : If node_color is a column name then a function or string giving the name of a function to aggregate that column for samples in each cluster.
* `node_cmap` : If node_color is 'samples' or a column name then a colourmap to use (see Colormap Matplotlib tutorial here: https://matplotlib.org/stable/tutorials/colors/colormaps.html).
* `edge_color` : For continuous colormap, use 'samples'. For discrete colors, use 'prefix' to color by resolution or specify a fixed color (see Specifying colors in Matplotlib tutorial here: https://matplotlib.org/stable/tutorials/colors/colors.html). If None, default set to 'samples'.
* `edge_cmap` : If edge_color is 'samples' then a colourmap to use (see Colormap Matplotlib tutorial here: https://matplotlib.org/stable/tutorials/colors/colormaps.html).
* `orientation` : Orientation of clustree drawing. Defaults to 'vertical'.
* `layout_reingold_tilford` : Whether to use the Reingold-Tilford algorithm for node positioning. Defaults to True if (kk <= 12), False otherwise. Setting True not recommended if (kk > 12) due to memory bottleneck in igraph dependency.
* `min_cluster_number` : Cluster number can take values (0, ..., K-1) or (1, ..., K). If the former option is preferred, parameter should take value 0, and 1 otherwise. Defaults to None, in which case, minimum cluster number is found automatically.
* `border_size` : Border width as proportion of image width. Defaults to 0.05.
* `figsize` : Parsed to matplotlib to determine figure size. Defaults to (kk/2, kk/2), clipped to a minimum of (3,3) and maximum of (10,10).
* `arrows` : Whether to add arrows to graph edges. Removing arrows alleviates appearance issue caused by arrows overlapping nodes. Defaults to True.
* `node_size` : Size of nodes in clustree graph drawing. Parsed directly to networkx.draw_networkx_nodes. Default to 300.
* `node_size_edge`: Controls edge start and end point. Parsed directly to networkx.draw_networkx_edges.
* `dpi` : Controls resolution of output if saved to file.
* `kk` : Choose custom depth of clustree graph.

## Glossary

* *cluster resolution*: Upper case `K`. For example, at cluster resolution `K=2` data is clustered into 2 distinct clusters.
* *cluster number*: Lower case `k`. For example, at cluster resolution 2 data is clustered into 2 distinct clusters `k=1` and `k=2`.
* *kk*: highest value of `K` (cluster resolution) shown in clustree.
* *cluster membership*: The association between data points and cluster numbers for fixed cluster resolution. For example, `[1, 1, 2, 2, 2]` would mean the first 2 data points belong to cluster number `1` and the following 3 data points belong to cluster number `2`.