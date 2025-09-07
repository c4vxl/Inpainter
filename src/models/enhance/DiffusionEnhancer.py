import torch
from PIL import Image

from diffusers import DiffusionPipeline # type: ignore

from config.Config import DIFFUSION_ENHANCER_DEFAULT_MODEL, LOAD_IN_4BIT, USE_SAFETY_CHECKER
from .base import Enhancer
from utils.image_utils import pad_to_size, remove_padding

class DiffusionEnhancer(Enhancer):
    def __init__(self, model_name: str = DIFFUSION_ENHANCER_DEFAULT_MODEL, load_in_4bit = LOAD_IN_4BIT, use_safety_checker: bool = USE_SAFETY_CHECKER, **kwargs) -> None:
        super().__init__()

        if load_in_4bit:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                load_in_4bit=True,
                **kwargs
            )
        else:
            self.pipe = DiffusionPipeline.from_pretrained(model_name, **kwargs)
        
        if not use_safety_checker:
            self.pipe.safety_checker = None
        
        self.pipe.enable_model_cpu_offload()


    def _generate(self, image: Image.Image, **kwargs) -> Image.Image:
        return self.generate(image, **kwargs)


    def generate(self, image: Image.Image, resolution: int = 512, prompt: str = "high quality, clean, sharp photo, no noise, no blur", **kwargs) -> Image.Image:
        image = pad_to_size(image, resolution)
        out = self.pipe(prompt=prompt, image=image, num_inference_steps=20, guidance_scale=6.5, **kwargs).images[0] # type: ignore
        out = remove_padding(out, resolution)
        return out