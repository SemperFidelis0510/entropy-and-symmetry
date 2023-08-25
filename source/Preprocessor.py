from Image import Image
import numpy as np
class Preprocessor:
    def __init__(self, crop_size=None):
        self.crop_size = crop_size

    def applyPreprocessing(self, image_object: Image):
        if self.crop_size is None:
            self.crop_size = min(image_object.rawData.size)
        cropped = image_object.rawData.crop((0, 0, self.crop_size, self.crop_size))
        img_arr = np.array(cropped)
        if img_arr.shape[2] == 4:
            img_arr = img_arr[:, :, :-1]
        elif img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)
        image_object.preprocessedData = img_arr