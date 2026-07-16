import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

class CustomRandomBlur:
    def __init__(self, p=0.5, max_radius=3):
        self.p = p
        self.max_radius = max_radius

    def __call__(self, img):
        if np.random.rand() < self.p:
            radius = np.random.uniform(0.5, self.max_radius)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img

class CustomRandomBrightnessContrast:
    def __init__(self, p=0.5, brightness_range=(0.5, 1.5), contrast_range=(0.5, 1.5)):
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, img):
        if np.random.rand() < self.p:
            factor_b = np.random.uniform(*self.brightness_range)
            img = ImageEnhance.Brightness(img).enhance(factor_b)
            factor_c = np.random.uniform(*self.contrast_range)
            img = ImageEnhance.Contrast(img).enhance(factor_c)
        return img

class CustomSaltPepperNoise:
    def __init__(self, p=0.5, amount=0.02):
        self.p = p
        self.amount = amount

    def __call__(self, img):
        if np.random.rand() < self.p:
            img_arr = np.array(img).copy()
            num_salt = np.ceil(self.amount * img_arr.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_arr.shape[:2]]
            img_arr[coords[0], coords[1]] = 255

            num_pepper = np.ceil(self.amount * img_arr.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_arr.shape[:2]]
            img_arr[coords[0], coords[1]] = 0
            
            return Image.fromarray(img_arr)
        return img