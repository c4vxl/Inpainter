from PIL import Image, ImageFilter

class Masker():
    def all_labels(self) -> dict[str, int]:
        raise NotImplementedError()

    def _compute_mask(self, image: Image.Image, segments = None) -> Image.Image:
        raise NotImplementedError()
    
    def _expand_mask(self, mask: Image.Image, pixels: int = 10) -> Image.Image:
        mask = mask.convert("L")

        for _ in range(pixels):
            mask = mask.filter(ImageFilter.MaxFilter(3))

        return mask
    
    def __call__(self, image: Image.Image, labels = None, expand_pixels: int = 0) -> Image.Image:
        mask = self._compute_mask(image, labels)
        mask = self._expand_mask(mask, expand_pixels)
        mask = mask.filter(ImageFilter.GaussianBlur(2))
        return mask