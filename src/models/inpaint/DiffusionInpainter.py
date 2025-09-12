import torch
from diffusers import DiffusionPipeline # type: ignore
from PIL import Image

from config.Config import DIFFUSION_INPAINTER_DEFAULT_MODEL, LOAD_IN_4BIT, USE_SAFETY_CHECKER
from .base import Inpainter

class DiffusionInpainter(Inpainter):
    def __init__(self, model_name: str = DIFFUSION_INPAINTER_DEFAULT_MODEL, load_in_4bit = LOAD_IN_4BIT, use_safety_checker: bool = USE_SAFETY_CHECKER, **kwargs) -> None:
        super().__init__()

        self.pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(model_name, load_in_4bit=load_in_4bit, torch_dtype=torch.bfloat16, **kwargs)
        
        if not use_safety_checker:
            self.pipe.safety_checker = None
        
        self.pipe.enable_model_cpu_offload()
    
    @torch.no_grad()
    def _generate(self, prompt: str, image: Image.Image, mask: Image.Image, **kwargs) -> list[Image.Image]:
        out = None
        
        # Gracefully remove optional arguments if generation fails
        optional_keys = ["reference_image", "negative_prompt", "mask_blur"]
        for _ in range(len(optional_keys) + 1):
            try:
                out = self.pipe(prompt=prompt, image=image, mask_image=mask, **kwargs)  # type: ignore
                break
            except TypeError as e:
                if not optional_keys:
                    raise
                kwargs.pop(optional_keys.pop(0), None)

        return out.images if out is not None else []
    
    def _load_lora_weights(self, path: str):
        self.pipe.load_lora_weights(path)