import numpy as np
from PIL import Image, ImageDraw, ImageFont


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
