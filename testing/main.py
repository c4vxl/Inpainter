import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from PIL import Image

from models.mask.SegformerB2Clothes import SegformerB2Clothes
from pipeline.PipeV1 import PipeV1

pipeline = PipeV1()

image = Image.open("images/input.png").convert("RGB")

images = pipeline(
    image = image,
    prompt = "Change to a blue shirt, T-Shirt, baggy clothing, blue",
    mask_labels = SegformerB2Clothes.SHIRT_LABELS,
    mask_expand = 30,
    guidance_scale = 10.,
    strength = 0.97,
    upscale_value = 4,
    fidelity = 0.8,
)

for img in images:
    img.show()