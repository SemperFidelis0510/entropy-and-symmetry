import platform
import subprocess
from io import BytesIO

import requests

from img_utils import *
from utils import *


def save_img(folder_path, images_arr):
    """
    Saves image arrays to the specified folder path as BMP files.

    Args:
        folder_path (str): Path to the folder where the images will be saved.
        images_arr (list or np.ndarray): List or array of image arrays to be saved.

    Note:
        If the folder path does not exist, it will be created.
        The function also attempts to open the folder using the default file explorer based on the OS.
    """
    if isinstance(images_arr, np.ndarray):
        images_arr = [images_arr]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    i = 0
    for arr in images_arr:
        path = os.path.join(folder_path, f'i={i}_s={arr[1]:.3f}.bmp')
        Image.fromarray(arr[0]).save(path)
        i += 1

    print(f'Sorted images saved to: {os.path.abspath(folder_path)}')
    if platform.system() == 'Windows':
        os.startfile(os.path.join(os.getcwd(), folder_path))
    elif platform.system() == 'Darwin' or platform.system() == 'Linux':
        subprocess.run(['open', os.path.join(os.getcwd(), folder_path)])
    else:
        print(f"Unsupported OS: {platform.system()}")


def load_images(path):
    """
    Loads image paths from a given directory, filtering for specific image file extensions.

    Args:
        path (str): Path to the directory containing the images.

    Returns:
        images_path (list): List of paths to the image files with extensions 'jpg', 'bmp', and 'png'.
    """
    images_path = []
    for filename in os.listdir(path):
        if filename.lower().endswith(('.jpg', '.bmp', '.png')):
            img_path = os.path.join(path, filename)
            images_path.append(img_path)
    return images_path


def preprocess(folder_path, crop_size=None, colors='rgb'):
    """
    Preprocesses images from a given folder path by cropping and converting to the specified color format.

    Args:
        folder_path (str): Path to the folder containing the images.
        crop_size (int, optional): Size of the cropped square. If None, the crop size will vary based on the image size.
        colors (str, optional): Color format, either 'rgb' or 'greyscale'. Default is 'rgb'.

    Returns:
        images_arr (list): List of preprocessed image arrays.
        paths (list): List of paths to the processed images.
    """
    if crop_size is None:
        vary_crop = True
    else:
        vary_crop = False

    paths = [p for p in load_images(folder_path) if p.lower().endswith(('.jpg', '.bmp', '.png'))]
    n = len(paths)
    i = 0

    images_arr = []
    start_time = time.time()
    for path in paths:
        i += 1

        img = Image.open(path)
        if colors == 'greyscale':
            img = img.convert('L')

        if vary_crop:
            crop_size = min(img.size)
        cropped = img.crop((0, 0, crop_size, crop_size))
        img_arr = np.array(cropped)

        if colors == 'rgb':
            if img_arr.ndim == 4:
                img_arr = img_arr[:, :, :-1]
            elif img_arr.ndim == 2:
                img_arr = np.stack([img_arr] * 3, axis=-1)

        images_arr.append(img_arr)
        print_progress_bar('Preprocessed', i, n, start_time=start_time)

    print(f'\nPreprocessing done.')

    return images_arr, paths


def get_google_map_image(latitude, longitude, zoom_level, width=500, height=500, save=False):
    """
    Retrieves a satellite image from Google Maps for the specified location, zoom level, and dimensions.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        zoom_level (int): Zoom level for the map image.
        width (int, optional): Width of the image in pixels. Default is 500.
        height (int, optional): Height of the image in pixels. Default is 500.
        save (bool, optional): Whether to save the image to a file. Default is False.

    Returns:
        image (np.ndarray): Array representing the retrieved image.

    Raises:
        Exception: If there is an error retrieving the image.
    """
    url = "https://maps.googleapis.com/maps/api/staticmap"
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    params = {
        "center": f"{latitude},{longitude}",
        "zoom": zoom_level,
        "size": f"{width}x{height}",
        "maptype": "satellite",
        "key": api_key
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image = image.crop((0, 0, image.width, image.height - 22))
        if save:
            image.save("../datasets/satellite/map_image.png")
        image = np.array(image)
        return image
    else:
        raise Exception(f"Error retrieving image: {response.text}")
