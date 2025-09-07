import torch.nn as nn
import numpy as np
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image

from config.Config import SEGFORMER_MASKER_DEFAULT_MODEL
from .base import Masker

class SegformerB2Clothes(Masker):
    CLOTHING_LABELS = ["Hat", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Scarf"]
    SHIRT_LABELS = ["Upper-clothes", "Dress", "Scarf"]
    PANTS_LABELS = ["Skirt", "Pants", "Dress", "Belt"]
    FACE_LABELS = ["Face", "Hair", "Sunglasses"]
    PERSON_LABELS = ["Face", "Hair", "Upper-clothes", "Skirt", "Pants", "Dress", "Belt", "Left-shoe", "Right-shoe", "Hat", "Left-leg", "Right-leg", "Left-arm", "Right-arm", "Bag", "Scarf", "Sunglasses" ]

    def __init__(self, model_name: str = SEGFORMER_MASKER_DEFAULT_MODEL) -> None:
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

        self.id2label = self.model.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

    def all_labels(self) -> dict[str, int]:
        return self.label2id

    def _compute_mask(self, image: Image.Image, segments = None) -> Image.Image:
        image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode="bilinear", align_corners=False)

        pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()

        if segments is None:
            mask = pred_seg.astype("uint8")
        else:
            selected_ids = [self.label2id[c] for c in segments if c in self.label2id]
            mask = np.isin(pred_seg, selected_ids).astype("uint8") * 255
        
        return Image.fromarray(mask)