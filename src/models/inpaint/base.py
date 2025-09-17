from PIL import Image, ImageFilter
import torch

from utils.image_utils import pad_to_size, remove_padding

class Inpainter():
    def _generate(self, prompt: str, image: Image.Image, mask: Image.Image, **kwargs) -> list[Image.Image]:
        raise NotImplementedError()

    def _load_lora_weights(self, path: str):
        raise NotImplementedError()

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        resolution: int = 2048,
        strict_mask: bool = False,
        strict_mask_forgiveness: int = 30,
        **kwargs
    ) -> list[Image.Image]:
        original_size = image.size
        image = pad_to_size(image, resolution)
        mask = pad_to_size(mask, resolution)
        out = self._generate(prompt, image, mask, **kwargs)

        if strict_mask and strict_mask_forgiveness > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=strict_mask_forgiveness))

        final = []
        for img in out:
            if strict_mask:
                m = mask.resize(img.size, Image.LANCZOS)
                i = image.resize(img.size, Image.LANCZOS)

                img = Image.composite(img, i, m)
            
            img = remove_padding(img, original_size)
            final.append(img)            

        return final