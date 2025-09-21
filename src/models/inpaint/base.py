from PIL import Image, ImageFilter
import torch
import numpy as np

from utils.image_utils import pad_to_size, remove_padding

class Inpainter():
    def _generate(self, prompt: str, image: Image.Image, mask: Image.Image, **kwargs) -> list[Image.Image]:
        raise NotImplementedError()

    def _load_lora_weights(self, path: str):
        raise NotImplementedError()

    def _expand_mask(self, mask: Image.Image, pixels: int = 10) -> Image.Image:
        mask = mask.convert("L")

        for _ in range(pixels):
            mask = mask.filter(ImageFilter.MaxFilter(3))

        return mask
    
    def _apply_strict_mask(self, original: Image.Image, generated: Image.Image, mask: Image.Image, forgiveness: int):
        original = original.resize(generated.size, Image.LANCZOS)
        mask = mask.resize(generated.size, Image.NEAREST)
        mask = mask.point(lambda x: 0 if x > 127 else 255).convert("1")
        
        generated = generated.resize(generated.size, Image.LANCZOS)

        generated = generated.convert("RGB")
        original = original.convert("RGB")
        mask = mask.convert("1")

        gen_np = np.array(generated)
        orig_np = np.array(original)
        mask_np = np.array(mask, dtype=bool)

        gen_np[mask_np] = orig_np[mask_np]

        return Image.fromarray(gen_np)

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
            mask = self._expand_mask(mask, strict_mask_forgiveness)
            mask = mask.point(lambda x: 255 if x > 127 else 0).convert("L")

        final = []
        for img in out:
            if strict_mask:
                img = self._apply_strict_mask(image, img, mask, strict_mask_forgiveness)
            
            img = remove_padding(img, original_size)
            final.append(img)            

        return final