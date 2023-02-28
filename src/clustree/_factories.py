import tempfile
from collections import defaultdict
from typing import Any, Callable, List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from pairing_functions import szudzik
from PIL import Image, ImageDraw, ImageFont

from clustree.clustree_typing import (
    EDGE_CONFIG_TYPE,
    IMAGE_CONFIG_TYPE,
    NODE_CMAP_TYPE,
    NODE_COLOR_AGG_TYPE,
    NODE_COLOR_TYPE,
    NODE_CONFIG_TYPE,
)

control_list = ["init", "sample_info", "image", "node_color", "draw"]
default_setup_config = {k: True for k in control_list}


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


class ClustreeConfig:
    def __init__(
        self,
        kk: int,
        data: pd.DataFrame,
        image_cf: IMAGE_CONFIG_TYPE,
        prefix: str,
        node_color: NODE_COLOR_TYPE = None,
        node_color_aggr: NODE_COLOR_AGG_TYPE = None,
        node_cmap: NODE_CMAP_TYPE = None,
        _setup_cf: Optional[dict[str, bool]] = None,
    ):
        if not node_color:
            node_color = prefix
        if not _setup_cf:
            _setup_cf = default_setup_config
        if not node_cmap:
            node_cmap = plt.cm.Blues

        self.prefix = prefix
        self.kk = kk
        self.node_cf: NODE_CONFIG_TYPE = defaultdict(dict)
        self.edge_cf: EDGE_CONFIG_TYPE = defaultdict(dict)
        self.k_upper_to_node_id: dict[int, list[int]] = {}
        self.node_color_cf: dict[str, Any] = dict()

        self.membership_cols = [
            f"{prefix}{str(k_upper)}" for k_upper in range(1, kk + 1)
        ]
        cluster_membership = data[self.membership_cols].to_numpy()

        if _setup_cf["init"]:
            self.init_cf()
        if _setup_cf["sample_info"]:
            self.set_sample_information(data=cluster_membership)
        if _setup_cf["image"]:
            self.set_image(image_cf=image_cf)
        if _setup_cf["node_color"]:
            self.set_node_color(
                node_color=node_color,
                aggr=node_color_aggr,
                cmap=node_cmap,
                prefix=prefix,
                data=data,
            )
        if _setup_cf["draw"]:
            self.set_drawn_image()

    @classmethod
    def hash_k_k(
        cls, k_upper: int, k_lower: Union[int, List[int]], k_start: Optional[int] = None
    ) -> Union[int, List[int]]:
        if k_start:
            return szudzik.pair(k_upper, k_lower, k_start)
        if isinstance(k_lower, list):
            return [szudzik.pair(k_upper, ele) for ele in k_lower]
        return szudzik.pair(k_upper, k_lower)

    def init_cf(self) -> None:
        for k_upper in range(1, self.kk + 1):
            for k_lower in range(1, k_upper + 1):
                ind = ClustreeConfig.hash_k_k(k_upper=k_upper, k_lower=k_lower)
                self.node_cf[ind].update(
                    {"k": k_lower, "res": k_upper}
                )
            self.k_upper_to_node_id[k_upper] = ClustreeConfig.hash_k_k(
                k_upper=k_upper, k_lower=list(range(1, k_upper + 1))
            )

    def set_image(self, image_cf: IMAGE_CONFIG_TYPE) -> None:
        for k_upper, v in image_cf.items():
            for k_lower, img in v.items():
                # convert str to int
                ind = ClustreeConfig.hash_k_k(
                    k_upper=int(k_upper), k_lower=int(k_lower)
                )
                self.node_cf[ind]["image"] = img

    def set_drawn_image(self) -> None:
        for k, v in self.node_cf.items():
            self.node_cf[k]["image_with_drawing"] = draw_circle(
                img=v["image"],
                radius=0.1,
                node_color=v["node_color"],
            )

    def set_sample_information(self, data: np.ndarray) -> None:
        """

        Parameters
        ----------
        data : ndarray
            Column 0 must be cluster membership for K = 1, and so on, finally column \
            (kk - 1) must be cluster membership for K = kk

        Returns
        -------
            None
        """
        for k_upper in range(1, self.kk + 1):
            col = k_upper - 1  # use (k_upper - 1) since data is 0-indexed
            vals, counts = np.unique(data[:, col], return_counts=True)
            for k_end, node_samples in zip(vals, counts):
                # get #samples at each node
                end_hashed = ClustreeConfig.hash_k_k(
                    k_upper=k_upper, k_lower=int(k_end)
                )
                self.node_cf[end_hashed]["samples"] = int(node_samples)

                if k_upper > 1:
                    # get #samples along each incoming edge
                    ind = data[:, col] == k_end
                    to_count = data[ind, col - 1]
                    vals, counts = np.unique(to_count, return_counts=True)
                    for k_start, edge_samples in zip(vals, counts):
                        start_hashed = ClustreeConfig.hash_k_k(
                            k_upper=k_upper - 1, k_lower=int(k_start)
                        )
                        self.edge_cf[
                            ClustreeConfig.hash_k_k(
                                k_upper=k_upper,
                                k_lower=int(k_end),
                                k_start=int(k_start),
                            )
                        ].update(
                            {
                                "alpha": (float(edge_samples) / float(node_samples)),
                                "samples": int(edge_samples),
                                "start": start_hashed,
                                "end": end_hashed,
                                "res": k_upper,
                            }
                        )

    def set_node_color(
        self,
        node_color: NODE_COLOR_TYPE,
        cmap: Optional[mpl.colors.Colormap],
        aggr: Optional[Callable],
        data: pd.DataFrame,
        prefix: str,
    ) -> None:
        if node_color == prefix:
            for node_id, attr in self.node_cf.items():
                self.node_cf[node_id]["node_color"] = f"C{attr['res']}"
        elif (use_samples := node_color == "samples") or (node_color in data.columns):
            # create to_parse = {node_id: value}
            if use_samples:
                to_parse = {k: v["samples"] for k, v in self.node_cf.items()}
            else:
                if not aggr:
                    raise ValueError(
                        "Cannot calculate node color without aggregate function"
                    )
                to_parse = {
                    ClustreeConfig.hash_k_k(k_upper=k_upper, k_lower=k_lower): float(
                        val
                    )
                    for k_upper, cluster_col in enumerate(self.membership_cols, 1)
                    for k_lower, val in data.groupby(cluster_col)[node_color]
                    .agg(aggr)
                    .to_dict()
                    .items()
                }

            # convert to_parse to {node_id: color}
            rgba, sm = _data_to_color(data=to_parse, cmap=cmap)
            for k, v in rgba.items():
                self.node_cf[k]["node_color"] = v
        else:  # fixed color, e.g., mpl.colors object
            for node_id in self.node_cf:
                self.node_cf[node_id]["node_color"] = node_color

def _data_to_color(
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


def get_fake_img(k_upper: str, k_lower: str, w: int = 40, h: int = 40) -> np.ndarray:
    """
    Create fake image reading 'K_k'.

    :param k_upper: cluster resolution
    :param k_lower: cluster number
    :param w: number of pixels
    :param h: number of pixels
    :return: image with white background and black text
    """
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    out = f"{k_upper}_{k_lower}"
    _, _, i, j = draw.textbbox((0, 0), out, font=font)
    draw.text(((w - i) / 2, (h - j) / 2), out, font=font, fill="black")
    return np.asarray(img)
