# clustree

## Status

**Functionality: Implemented**

* Directed graph representing clustree. Nodes are parsed images and node information is encoded by a small circle in the corner of the image.
* Data and images provided directly or through a path to parent directory.
* If parsed directly, data should be a `pd.DataFrame` object.
* If parsed directly, images should be a dictionary. See more information in Quickstart.
* Edge and node color can correspond to one of: #samples that pass through edge/node, cluster resolution `K`, or a fixed color. In the case of node color, a column name in the data and aggregate function can be used too. Use of column name and #samples creates a continuous colormap, whilst the other options result in discrete colors.


**Functionality: To Add**

* Legend for continuous colormaps.
* Reingold-Tilford algorithm to minimise crossing edges.
* Allow PDF inputs.
* Much more! Early testing will help prioritise future development.

## Usage

### Installation

TO ADD.

### Quickstart

The powerhouse function of the library is `clustree`. See details on its usage below.

```
def clustree(
    data: Union[str, Path, pd.DataFrame],
    prefix: str,
    images: Union[str, Path, dict[int, np.ndarray]],
    output_path: Optional[Union[str, Path]] = None,
    draw: bool = True,
    node_color: Any = None,
    node_color_aggr: Optional[Union[Callable, str]] = None,
    node_cmap: Optional[Union[mpl.colors.Colormap, str]] = None,
    edge_color: Any = None,
    edge_cmap: Optional[Union[mpl.colors.Colormap, str]] = None,
    errors: bool = False,
) -> DiGraph:
```

* `data`: Path of csv or DataFrame object.
* `prefix`: String indicating columns containing clustering information.
* `images`: String indicating directory containing images. See more information on files expected in directory in Notes.
* `output_path`: Directory to output the final plot to. If None, then output not wrriten to file.
* `draw`: Whether to draw the clustree. Defaults to True. If False and output_path supplied, will be overridden. Saving to file requires drawing.
* `node_color`: For continuous colormap, use 'samples' or the name of a metadata column to color nodes by. For discrete colors, use 'prefix' to color by resolution or specify a fixed color (see Specifying colors in Matplotlib tutorial here: https://matplotlib.org/stable/tutorials/colors/colors.html).
* `node_color_aggr`: If node_color is a column name then a function or string giving the name of a function to aggregate that column for samples in each cluster.
* `node_cmap`: If node_color is 'samples' or a column name then a colourmap to use (see Colormap Matplotlib tutorial here: https://matplotlib.org/stable/tutorials/colors/colormaps.html).
* `edge_color`: For continuous colormap, use 'samples'. For discrete colors, use 'prefix' to color by resolution or specify a fixed color (see Specifying colors in Matplotlib tutorial here: https://matplotlib.org/stable/tutorials/colors/colors.html).
* `edge_cmap`: If edge_color is 'samples' then a colourmap to use (see Colormap Matplotlib tutorial here: https://matplotlib.org/stable/tutorials/colors/colormaps.html).
* `errors`: Whether to raise an error if an image is missing from directory supplied to images parameter. If False, a fake image will be created with text 'K_k' where K is cluster resolution and k is cluster number. Defaults to False.

## Glossary

* *cluster resolution*: Upper case `K`. For example, at cluster resolution `K=2` data is clustered into 2 distinct clusters.
* *cluster number*: Lower case `k`. For example, at cluster resolution 2 data is clustered into 2 distinct clusters `k=1` and `k=2`.
* *kk*: highest value of `K` (cluster resolution) shown in clustree.
* *cluster membership*: The association between data points and cluster numbers for fixed cluster resolution. For example, `[1, 1, 2, 2, 2]` would mean the first 2 data points belong to cluster number `1` and the following 3 data points belong to cluster number `2`.

## Contributing

* python -m flake8 src --max-line-length=88
* python -m flake8 tests --max-line-length=88
* python -m pytest .
* black .
* isort .
