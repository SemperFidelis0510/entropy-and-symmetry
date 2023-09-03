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
            self.crop_size = min(image_object.rawData.size(-2),
                                 image_object.rawData.size(-1))  # Assuming shape is (C, H, W)

            # Cropping using PyTorch, assuming shape is (C, H, W)
        cropped = image_object.rawData[:, :self.crop_size, :self.crop_size]

        # If the tensor has 4 channels, remove the last one
        if cropped.size(0) == 4:
            cropped = cropped[:3, :, :]

        # If the tensor is grayscale (1 channel), repeat it to make it 3 channels
        elif cropped.size(0) == 1:
            cropped = cropped.repeat(3, 1, 1)

        image_object.preprocessedData = cropped

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
                # Scale and shift to 8-bit integer values
                y = np.clip((219 * y + 16), 16, 235).astype(np.uint8)
                cb = np.clip((224 * cb + 128), 16, 240).astype(np.uint8)
                cr = np.clip((224 * cr + 128), 16, 240).astype(np.uint8)
                ycbcr_image_array = np.stack([y, cb, cr], axis=-1)
                img = ycbcr_image_array
            case 'greyscale':
                img = rgb2gray(img)

        return img
