import numpy as np
from PIL import Image as PILImage
import os
from Image import Image
import time
from utils import print_progress_bar
class ImageLoader:
    def __init__(self, image_directory, image_format=('png'), head=None):
        self.image_directory = image_directory
        self.image_format = image_format
        self.image_paths = []
        self.head = head
        self.prepare_path()
    def load_images(self):
        image_objects = []
        self.prepare_path()
        start_time = time.time()
        n = len(self.image_paths)
        for index, filename in enumerate(self.image_paths):
            img_data = PILImage.open(filename).convert('RGB')
            image_object = Image(img_data, filename)
            image_objects.append(image_object)
            print_progress_bar('Loading Image', index+1, n, start_time=start_time)
        print(f'\nLoading done. Please wait for entropy calculation to start.')
        return image_objects

    def load_batch_images(self, batch_size):
        image_objects = []
        start_time = time.time()
        n = len(self.image_paths)
        for index, filename in enumerate(self.image_paths):
            img_data = PILImage.open(filename).convert('RGB')
            image_object = Image(img_data, filename)  # Assuming Image is your custom class
            image_objects.append(image_object)
            print_progress_bar('Loading Image', index + 1, n, start_time=start_time)

            if len(image_objects) == batch_size:
                yield image_objects
                image_objects = []

        if image_objects:  # yield any remaining images
            yield image_objects

    def prepare_path(self):
        if os.path.isdir(self.image_directory):
            for root, _, filenames in os.walk(self.image_directory):
                for filename in filenames:
                    if filename.lower().endswith(self.image_format):
                        img_path = os.path.join(root, filename)
                        self.image_paths.append(img_path)
        elif os.path.isfile(self.image_directory) and self.image_directory.lower().endswith(self.image_format):
            self.image_paths = [self.image_directory]
        else:
            raise ValueError("The provided path is neither a directory nor a valid image file.")

        if self.head is not None:
            self.image_paths = self.image_paths[:self.head]

