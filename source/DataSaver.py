import os
import platform
from Image import Image
from PIL import Image as PILImage
import subprocess
class DataSaver:
    def __init__(self, destination):
        self.destination = destination
        self.format = format

    def save(self, index, image):
        if not os.path.exists(self.destination):
            os.makedirs(self.destination)
        path = os.path.join(self.destination, f'i={index}.bmp')
        image.rawData.save(path)
