from PIL import Image

from models.mask.SegformerB2Clothes import SegformerB2Clothes
from models.inpaint.DiffusionInpainter import DiffusionInpainter
from models.enhance.CodeFormer import CodeFormer
import config.Config as Config

from .base import Pipeline

class PipeV1(Pipeline):
    def __init__(self,
                 load_in_4bit: bool = Config.LOAD_IN_4BIT, use_safety_checker: bool = Config.USE_SAFETY_CHECKER,
                 masking_model: str = Config.SEGFORMER_MASKER_DEFAULT_MODEL, inpainter_model: str = Config.DIFFUSION_INPAINTER_DEFAULT_MODEL,) -> None:
        self.masker = SegformerB2Clothes(masking_model)
        self.inpainter = DiffusionInpainter(inpainter_model, load_in_4bit=load_in_4bit, use_safety_checker=use_safety_checker)
        self.enhancer = CodeFormer("cuda" if "torch.cuda.is_available()" else "cpu")

    def _generate_mask(self, image: Image.Image, labels: list[str], expand: int = 0) -> Image.Image:
        return self.masker(image, labels, expand)

    def _enhance_images(self, images: list[Image.Image],
                        upscale_value: int = 2, enhance_background: bool = True, face_upsample: bool = True, fidelity: float = 0.8) -> list[Image.Image]:
        return self.enhancer(
            images, upscale = upscale_value,
            background_enhance = enhance_background, face_upsample = face_upsample,
            codeformer_fidelity = fidelity
        )
    
    def _generate_images(self, image: Image.Image, mask: Image.Image,
                  prompt: str, negative_prompt: str = "",
                  resolution: int = 2048, guidance_scale: float = 10., strength: float = 0.4,
                  mask_blur: int = 1, num_inference_steps: int = 75, num_images_per_prompt: int = 4):
        return self.inpainter(
            prompt, image, mask, resolution,
            reference_image=image,
            guidance_scale=guidance_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
            mask_blur=mask_blur,
            negative_prompt = negative_prompt,
            num_images_per_prompt=num_images_per_prompt
        )

    def _generate(self, image: Image.Image, prompt: str, mask_labels: list[str], mask_expand: int = 0,
                  negative_prompt: str = "",
                  resolution: int = 2048, guidance_scale: float = 10., strength: float = 0.4,
                  mask_blur: int = 1, num_inference_steps: int = 75, num_images_per_prompt: int = 4,
                  upscale_value: int = 2, enhance_background: bool = True, face_upsample: bool = True, fidelity: float = 0.8) -> list[Image.Image]:
        mask = self._generate_mask(image, mask_labels, mask_expand)
        images = self._generate_images(image, mask, prompt, negative_prompt, resolution, guidance_scale, strength, mask_blur, num_inference_steps, num_images_per_prompt)
        enhanced = self._enhance_images(images, upscale_value, enhance_background, face_upsample, fidelity)

        return enhanced