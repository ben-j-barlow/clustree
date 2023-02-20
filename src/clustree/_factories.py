from collections import defaultdict
from typing import Optional, Union
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import pandas as pd
from pairing_functions import szudzik

from clustree.clustree_typing import (
    EDGE_CONFIG_TYPE,
    IMAGE_CONFIG_TYPE,
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
        _setup_cf: Optional[dict[str, bool]] = None,
    ):
        self.prefix = prefix
        self.kk = kk
        self.node_cf: NODE_CONFIG_TYPE = defaultdict(dict)
        self.edge_cf: EDGE_CONFIG_TYPE = defaultdict(dict)

        if not _setup_cf:
            _setup_cf = default_setup_config

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
            self.set_color()

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
            for k_lower, node_samples in zip(vals, counts):
                # get #samples at each node
                k_lower = int(k_lower)
                self.node_cf[ClustreeConfig.hash_k_k(k_upper=k_upper, k_lower=k_lower)][
                    "samples"
                ] = int(node_samples)

                if k_upper > 1:
                    # get #samples along each incoming edge
                    ind = data[:, col] == k_lower
                    to_count = data[ind, col - 1]
                    vals, counts = np.unique(to_count, return_counts=True)
                    for k_start, edge_samples in zip(vals, counts):
                        self.edge_cf[
                            ClustreeConfig.hash_k_k(
                                k_upper=k_upper, k_lower=k_lower, k_start=int(k_start)
                            )
                        ].update(
                            {
                                "alpha": (float(edge_samples) / float(node_samples)),
                                "samples": int(edge_samples),
                                "k_start_upper": k_upper - 1,
                                "k_start_lower": k_start,
                                "k_end_upper": k_upper,
                                "k_end_lower": k_lower,
                            }
                        )

    def set_color(self) -> None:
        pass


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
