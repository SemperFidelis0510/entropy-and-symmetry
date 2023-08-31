from source.Image import Image
import numpy as np
import colorsys
from skimage.color import rgb2gray
class Preprocessor:
    def __init__(self, crop_size=None, channels='rgb'):
        self.crop_size = crop_size
        self.channels = channels
    def applyPreprocessing(self, image_object: Image):
        if self.crop_size is None:
            self.crop_size = min(image_object.rawData.size)
        cropped = image_object.rawData.crop((0, 0, self.crop_size, self.crop_size))
        img_arr = np.array(cropped)
        if img_arr.shape[2] == 4:
            img_arr = img_arr[:, :, :-1]
        elif img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        image_object.preprocessedData = self.change_channels(img_arr)

    def change_channels(self, img):
        match self.channels:
            case 'rgb':
                img = img
            case 'hsb':
                # Normalize RGB values to [0, 1]
                img = img / 255.0

                # Initialize an empty array for HSB values
                hsb_image = np.zeros_like(img, dtype=float)

                # Iterate through the pixels and convert RGB to HSB
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        r, g, b = img[i, j]
                        h, s, v = colorsys.rgb_to_hsv(r, g, b)
                        hsb_image[i, j] = (h * 360, s * 100, v * 100)  # Convert to degrees and percentage

                img = hsb_image

            case 'YCbCr':
                kR = 0.299
                kG = 0.587
                kB = 0.114
                R = img[..., 0]
                G = img[..., 1]
                B = img[..., 2]
                y = kR * R + kG * G + kB * B
                cb = (-kR / (2 * (1 - kB))) * R + (-kG / (2 * (1 - kB))) * G + 1 / 2 * B
                cr = 1 / 2 * R + (-kG / (2 * (1 - kR))) * G + (-kB / (2 * (1 - kR))) * B
                ycbcr_image_array = np.stack([y, cb, cr], axis=-1)
                img = ycbcr_image_array
            case 'greyscale':
                img = rgb2gray(img)

        return img
