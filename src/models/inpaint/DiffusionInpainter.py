import torch
from diffusers import DiffusionPipeline # type: ignore
from PIL import Image

from config.Config import DIFFUSION_INPAINTER_DEFAULT_MODEL, LOAD_IN_4BIT, USE_SAFETY_CHECKER
from .base import Inpainter

class DiffusionInpainter(Inpainter):
    def __init__(self, model_name: str = DIFFUSION_INPAINTER_DEFAULT_MODEL, load_in_4bit = LOAD_IN_4BIT, use_safety_checker: bool = USE_SAFETY_CHECKER, **kwargs) -> None:
        super().__init__()

        if load_in_4bit:
            self.pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                load_in_4bit=True,
                **kwargs
            )
        else:
            self.pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(model_name, **kwargs)
        
        if not use_safety_checker:
            self.pipe.safety_checker = None
        
        self.pipe.enable_model_cpu_offload()
    
    def _generate(self, prompt: str, image: Image.Image, mask: Image.Image, **kwargs) -> list[Image.Image]:
        out = self.pipe(prompt=prompt, image=image, mask_image=mask, **kwargs) # type: ignore

        return out.images
    
    def _load_lora_weights(self, path: str):
        self.pipe.load_lora_weights(path)