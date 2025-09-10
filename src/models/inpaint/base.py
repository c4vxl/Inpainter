from PIL import Image
import torch

from utils.image_utils import pad_to_size, remove_padding

class Inpainter():
    def _generate(self, prompt: str, image: Image.Image, mask: Image.Image, **kwargs) -> list[Image.Image]:
        raise NotImplementedError()

    def _load_lora_weights(self, path: str):
        raise NotImplementedError()

    @torch.no_grad()
    def __call__(self, prompt: str, image: Image.Image, mask: Image.Image, resolution = 2048, **kwargs) -> list[Image.Image]:
        original_size = image.size
        image = pad_to_size(image, resolution)
        mask = pad_to_size(mask, resolution)
        out = self._generate(prompt, image, mask, **kwargs)
        return [ remove_padding(img, original_size) for img in out ]