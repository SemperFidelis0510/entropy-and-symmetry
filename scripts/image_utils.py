import numpy as np
from PIL import Image, ImageTk

def resize(input_array: np.ndarray, desired_size: tuple) -> np.ndarray:
    """
    Resize the input image (numpy array) to the desired size.

    Parameters:
    - input_array (np.ndarray): The input image as a numpy array.
    - desired_size (tuple): The desired size as (width, height).

    Returns:
    - np.ndarray: The resized image as a numpy array.
    """
    image = Image.fromarray(input_array)
    resized_image = image.resize(desired_size, Image.ANTIALIAS)
    return np.array(resized_image)

if __name__ == "__main__":
    # Example usage
    input_array = np.array(Image.open("path_to_your_image.jpg"))
    resized_array = resize_image_from_array(input_array, (500, 500))  # Example size 500x500
    Image.fromarray(resized_array).show()