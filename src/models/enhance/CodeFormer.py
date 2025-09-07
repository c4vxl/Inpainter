from typing import Optional
import os
from platformdirs import user_cache_dir

import cv2
from PIL import Image

import torch
import numpy as np

from torchvision.transforms.functional import normalize

from codeformer.basicsr.archs.rrdbnet_arch import RRDBNet
from codeformer.basicsr.utils import img2tensor, tensor2img
from torch.hub import download_url_to_file
from codeformer.basicsr.utils.realesrgan_utils import RealESRGANer
from codeformer.basicsr.utils.registry import ARCH_REGISTRY

from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from codeformer.facelib.utils.misc import is_gray

from .base import Enhancer
from utils.image_utils import as_cv2_img, as_pil_img

class CodeFormer(Enhancer):
    pretrain_model_url = {
        'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
        'realesrgan': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'
    }
    
    def __init__(self, device: str) -> None:
        self.device = device

        # Prepare model states
        self.model_paths = self._get_model_paths(user_cache_dir("CodeFormer"))
        self._prepare_models()

        # Prepare upsampler
        self.upsampler = self._create_upsampler()

        # Initialize codeformer net
        self.codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)

        # Load codeformer state
        self.codeformer_net.load_state_dict(torch.load(self.model_paths["codeformer"])["params_ema"])
        self.codeformer_net.eval()

    def _get_model_paths(self, cache_dir: str):
        return {
            'codeformer': os.path.join(cache_dir, 'CodeFormer', 'codeformer.pth'),
            'detection': os.path.join(cache_dir, 'facelib', 'detection_Resnet50_Final.pth'),
            'parsing': os.path.join(cache_dir, 'facelib', 'parsing_parsenet.pth'),
            'realesrgan': os.path.join(cache_dir, 'realesrgan', 'RealESRGAN_x2plus.pth')
        }

    def _prepare_models(self):
        for name, path in self.model_paths.items():
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                download_url_to_file(CodeFormer.pretrain_model_url[name], path, hash_prefix=None, progress=True)

    def _create_upsampler(self):
        # half = True if torch.cuda.is_available() else False
        half = True if torch.cuda.is_available() else False
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        upsampler = RealESRGANer(
            scale=2,
            model_path=self.model_paths["realesrgan"],
            model=model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=half,
        )
        return upsampler
    
    def _generate(self, image: Image.Image, **kwargs) -> Image.Image:
        return self.generate(image, **kwargs) # type: ignore

    def generate(self,
                 image: Image.Image, upscale: int = 2, background_enhance: bool = True, face_upsample: bool = True, codeformer_fidelity: float = 0.5,
                 has_aligned = False, draw_box = False, detection_model = "retinaface_resnet50"
                ) -> Optional[Image.Image]:
        # limit upscale based on memory and image size
        height = image.size[1]
        upscale = 2 if upscale > 2 and height > 1000 else max(upscale, 4)
        if height > 1500:
            upscale = 1
            background_enhance = face_upsample = False
        
        # Convert to cv2 image
        img = as_cv2_img(image)

        # Initialize face helper
        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
            device=self.device,
        )

        # Pick upsampler
        bg_upsampler = self.upsampler if background_enhance else None
        face_upsampler = self.upsampler if face_upsample else None

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            face_helper.align_warp_face() # align and warp each face
        
        # face restoration for each cropped face
        for cropped_face in face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True) # type: ignore
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device) # type: ignore

            try:
                with torch.no_grad():
                    output = self.codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype("uint8") # type: ignore
            face_helper.add_restored_face(restored_face)
        
        # upsample the background
        if bg_upsampler is not None:
            # Now only support RealESRGAN for upsampling background
            bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
        else:
            bg_img = None
        
        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )

            return as_pil_img(restored_img)