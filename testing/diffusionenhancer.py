import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from PIL import Image

from models.enhance.DiffusionEnhancer import DiffusionEnhancer

enhancer = DiffusionEnhancer("stabilityai/stable-diffusion-x4-upscaler")

image = Image.open("images/input.png")

image.show("Before")

out = enhancer([image])

for img in out:
    img.show("After")