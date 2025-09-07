from PIL import Image
import torch
import torchvision.transforms as T

from utils.image_utils import pad_to_size, remove_padding

class Enhancer():
    def __init__(self) -> None:
        self.transform = T.Compose([
            T.ToTensor(),                # converts to [0,1], shape [C,H,W]
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # scale to [-1,1]
        ])

    def _generate(self, image: Image.Image, **kwargs) -> Image.Image:
        raise NotImplementedError()

    @torch.no_grad()
    def __call__(self, images: list[Image.Image], **kwargs) -> list[Image.Image]:
        out = []
        for img in images:
            out.append(self._generate(img, **kwargs))
        
        return out