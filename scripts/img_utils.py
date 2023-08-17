import numpy as np
from PIL import Image


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


def extract_image_part(image, row_percentage, col_percentage, size_percentage):
    rows, cols, _ = image.shape
    size = int(min(rows, cols) * size_percentage / 100)

    # # Ensure the size is within bounds
    # size = min(size, rows, cols)

    row_start = int(rows * row_percentage / 100)
    col_start = int(cols * col_percentage / 100)

    # Adjust the starting coordinates if they would cause the extracted part to exceed the image bounds
    row_start = min(row_start, rows - size)
    col_start = min(col_start, cols - size)

    row_end = row_start + size
    col_end = col_start + size

    return image[row_start:row_end, col_start:col_end]
