import numpy as np
from PIL import Image as PILImage
import os
from source.Image import Image
import time
from source.utils import print_progress_bar
import torch
from torchvision import transforms
class ImageLoader:
    def __init__(self, callback=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if callback is None:
            self.print_progress_bar = print_progress_bar
        else:
            self.print_progress_bar = callback
    def load_images(self, image_paths, base_index):
        image_objects = []
        start_time = time.time()
        n = len(image_paths)

        # Define a transform to convert PIL images to PyTorch tensors
        to_tensor = transforms.ToTensor()

        for index, filename in enumerate(image_paths):
            img_data = PILImage.open(filename).convert('RGB')

            # Convert PIL image to PyTorch tensor and move it to the device
            img_data = to_tensor(img_data).to(self.device)

            image_index = index + base_index
            image_object = Image(img_data, filename, image_index)
            image_objects.append(image_object)

            self.print_progress_bar('Loading Image', index + 1, n, start_time=start_time)

        print(f'\nLoading done. Please wait for entropy calculation to start.')
        return image_objects


