from PIL import Image

class Pipeline():    
    def __call__(self, image: Image.Image, **kwargs) -> list[Image.Image]:
        return self._generate(image, **kwargs) # type: ignore