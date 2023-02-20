from collections import defaultdict
from typing import Any, Optional, Union

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from pairing_functions import szudzik
from PIL import Image, ImageDraw, ImageFont
from enum import Enum

from clustree.clustree_typing import (
    EDGE_CONFIG_TYPE,
    IMAGE_CONFIG_TYPE,
    NODE_COLOUR_TYPE,
    NODE_CONFIG_TYPE,
)

default_setup_config = {"init": True, "sample_info": True, "image": True, "color": True}


class ClustreeConfig:
    def __init__(
        self,
        kk: int,
        data: pd.DataFrame,
        image_cf: IMAGE_CONFIG_TYPE,
        prefix: str,
        node_colour: NODE_COLOUR_TYPE = None,
        _setup_cf: Optional[dict[str, bool]] = None,
    ):
        if not node_colour:
            # TODO: change to "res" once implemented
            node_colour = "samples"
        if not _setup_cf:
            _setup_cf = default_setup_config


        self.prefix = prefix
        self.kk = kk
        self.node_cf: NODE_CONFIG_TYPE = defaultdict(dict)
        self.edge_cf: EDGE_CONFIG_TYPE = defaultdict(dict)
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
        if _setup_cf["color"]:
            self.set_node_color(node_colour=node_colour, prefix=prefix, data=data)

    @classmethod
    def hash_k_k(cls, k_upper: int, k_lower: int, k_start: Optional[int] = None) -> int:
        if k_start:
            return szudzik.pair(k_upper, k_lower, k_start)
        return szudzik.pair(k_upper, k_lower)

    def init_cf(self) -> None:
        for k_upper in range(1, self.kk + 1):
            for k_lower in range(1, k_upper + 1):
                ind = ClustreeConfig.hash_k_k(k_upper=k_upper, k_lower=k_lower)
                self.node_cf[ind].update({"k_lower": k_lower, "k_upper": k_upper})
                self.edge_cf[ind].update({"k_lower": k_lower, "k_upper": k_upper})

    def set_image(self, image_cf: IMAGE_CONFIG_TYPE) -> None:
        for k_upper, v in image_cf.items():
            for k_lower, img in v.items():
                # convert str to int
                k_upper = int(k_upper)
                k_lower = int(k_lower)
                self.node_cf[ClustreeConfig.hash_k_k(k_upper=k_upper, k_lower=k_lower)][
                    "image"
                ] = img

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
                            }
                        )

    def set_node_color(
        self, node_colour: NODE_COLOUR_TYPE, data: pd.DataFrame, prefix: str
    ) -> None:
        if node_colour == prefix:
            self.node_color_cf["method"] = NodeColourMethod.RES
            raise NotImplementedError()
        elif node_colour == "samples":
            to_parse = {k: v["samples"] for k, v in self.node_cf.items()}
            rgba, sm = _data_to_rgba(data=to_parse)

            for_update = {k: {"node_color": v} for k, v in rgba.items()}
            self.node_cf.update(for_update)
            self.node_color_cf["method"] = NodeColourMethod.SAMPLE_CNT

        elif node_colour in data.columns:
            self.node_color_cf["method"] = NodeColourMethod.COL_NAME
            raise NotImplementedError()
        else:  # fixed color, e.g., mpl.colors object
            raise NotImplementedError()


class NodeColourMethod(Enum):
    FIXED = 1
    RES = 2
    SAMPLE_CNT = 3
    COL_NAME = 4


def _data_to_rgba(
    data: dict[int, Union[int, float]], cmap: mpl.colors.Colormap = mpl.cm.Blues
) -> tuple[dict[int, tuple[float, float, float, float]], ScalarMappable]:
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
    return {k: sm.to_rgba(v) for k, v in data.items()}, sm


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
