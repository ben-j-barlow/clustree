import tempfile
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from PIL import Image, ImageDraw, ImageFont


def draw_circle(
    img: np.ndarray,
    radius: float = 0.1,
    node_color: str = "red",
    img_width: int = 100,
    img_height: int = 100,
) -> np.ndarray:
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, img_width, 0, img_height])

    # Calculate radius and coordinates
    radius *= img_width
    x0 = img_width - radius
    y0 = img_height - radius

    # Add a circle at the top right of the image
    circle = plt.Circle((x0, y0), radius=radius, fill=True, color=node_color)
    ax.add_artist(circle)

    # save and read in
    fig.patch.set_visible(False)
    ax.axis("off")
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        file_path = f.name
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        to_return = plt.imread(file_path)
        plt.close()
        return to_return


def _data_to_color(
    data: dict[int, Union[int, float]],
    cmap: mpl.colormaps = mpl.cm.Blues,
    return_sm: bool = True,
) -> Union[
    dict[int, tuple[float, float, float, float]],
    tuple[dict[int, tuple[float, float, float, float]], ScalarMappable],
]:
    """

    Parameters
    ----------
    data
        Node / edge id as key and value to use for RGBA mapping as value. For example, \
        if determining RGBA value for node_color, value of dict could be #samples.
    cmap
        Colormap to use for int to RGBA mapping.

    Returns
    -------
        The keys (node or edge key) and RGBA values.

        The ScalarMappable object to allow colorbar visualization at plot time.
    """
    val = data.values()
    norm = mpl.colors.Normalize(vmin=min(val), vmax=max(val))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    if return_sm:
        return {k: sm.to_rgba(v) for k, v in data.items()}, sm
    return {k: sm.to_rgba(v) for k, v in data.items()}
