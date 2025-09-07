from PIL import Image, ImageDraw
import numpy as np
import cv2

def pad_to_size(image: Image.Image, size: int|tuple[int,int] = 1024):
    size = (size, size) if isinstance(size, int) else size

    # Scale image
    aspect_ratio = image.width / image.height
    new_width = size[0] if image.width > image.height else int(size[1] * aspect_ratio)
    new_height = int(size[0] / aspect_ratio) if image.width > image.height else size[1]

    # Paste scaled version
    out = Image.new(image.mode, size)
    out.paste(image.resize((new_width, new_height)), ((size[0] - new_width) // 2, (size[1] - new_height) // 2))

    return out

def remove_padding(padded_image: Image.Image, original_size: int | tuple[int, int] = 1024) -> Image.Image:
    original_size = (original_size, original_size) if isinstance(original_size, int) else original_size
    original_width, original_height = original_size

    # Calculate original aspact ratio
    padded_width, padded_height = padded_image.size
    original_aspect = original_width / original_height

    # Calculate new positions
    new_width = int(padded_height * original_aspect) if padded_width / padded_height > original_aspect else padded_width
    new_height = padded_height if padded_width / padded_height > original_aspect else int(padded_width / original_aspect)

    left = (padded_width - new_width) // 2
    top = (padded_height - new_height) // 2

    # Crop padding
    return padded_image.crop((left, top, left + new_width, top + new_height))

def as_cv2_img(image: Image.Image) -> np.ndarray:
    data = np.array(image.convert("RGB"))
    out = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return out

def as_pil_img(image: np.ndarray) -> Image.Image:
    data = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out = Image.fromarray(data)
    return out

def compose_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    mask_color = (68, 123, 160, 200)  # #447ba067 RGBA
    mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_layer)
    mask_draw.bitmap((0, 0), mask, fill=mask_color)
    return Image.alpha_composite(image.convert("RGBA"), mask_layer)