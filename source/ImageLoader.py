import numpy as np
from PIL import Image as PILImage
import os
from source.Image import Image
import time
from source.utils import print_progress_bar


class ImageLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_images(image_paths, base_index):
        image_objects = []
        start_time = time.time()
        n = len(image_paths)
        for index, filename in enumerate(image_paths):
            img_data = PILImage.open(filename).convert('RGB')
            image_index = index + base_index
            image_object = Image(img_data, filename, image_index)
            image_objects.append(image_object)
            print_progress_bar('Loading Image', index + 1, n, start_time=start_time)
        print(f'\nLoading done. Please wait for entropy calculation to start.')
        return image_objects

