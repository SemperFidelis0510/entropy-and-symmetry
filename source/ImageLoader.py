from PIL import Image as PILImage
import os
from Image import Image
class ImageLoader:
    def __init__(self, image_directory, image_format=('png')):
        self.image_directory = image_directory
        self.image_format = image_format
        self.image_paths = []
    def load_images(self):
        # image_objects = []
        # for filename in os.listdir(self.image_directory):
        #     if filename.endswith(self.image_format):
        #         img_path = os.path.join(self.image_directory, filename)
        #         img_data = PILImage.open(img_path)
        #         image_object = Image(img_data)
        #         image_objects.append(image_object)
        # return image_objects
        self.normalize_path()
        if os.path.isdir(self.image_directory):
            images_path = []
            for root, _, filenames in os.walk(self.image_directory):
                for filename in filenames:
                    if filename.lower().endswith(('.jpg', '.bmp', '.png')):
                        img_path = os.path.join(root, filename)
                        images_path.append(img_path)
            return images_path
        elif os.path.isfile(self.image_directory) and self.image_directory.lower().endswith(('.jpg', '.bmp', '.png')):
            return [self.image_directory]
        else:
            raise ValueError("The provided path is neither a directory nor a valid image file.")

    def normalize_path(self):
        if os.path.isdir(self.image_directory):
            for root, _, filenames in os.walk(self.image_directory):
                for filename in filenames:
                    if filename.lower().endswith(self.image_format):
                        img_path = os.path.join(root, filename)
                        self.image_paths.append(img_path)
        elif os.path.isfile(self.image_directory) and self.image_directory.lower().endswith(self.image_format):
            self.image_directory = [self.image_directory]
