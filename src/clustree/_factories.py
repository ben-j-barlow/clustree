from PIL import Image, ImageDraw, ImageFont


def _get_fake_img(k_upper: str, k_lower: str) -> Image:
    W, H = 40, 40
    img = Image.new("RGB", (W, H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    out = f"{k_upper}_{k_lower}"
    _, _, w, h = draw.textbbox((0, 0), out, font=font)
    draw.text(((W - w) / 2, (H - h) / 2), out, font=font, fill="black")
    return img
