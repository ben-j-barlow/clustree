from typing import Union

import matplotlib as mpl
from matplotlib.cm import ScalarMappable


def data_to_color(
    data: dict[int, Union[int, float]],
    cmap: mpl.colors.Colormap = mpl.cm.Blues,
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
