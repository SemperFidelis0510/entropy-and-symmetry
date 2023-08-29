import platform
import random
import json
import subprocess
from io import BytesIO

import requests

from entropy import *
from img_utils import *
import json


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
        path = os.path.join(folder_path, f'i={i}_s={arr[1]}.bmp')
        img = arr[0]
        Image.fromarray(img, 'RGB').save(path)
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
    if isinstance(path, list):
        path = normalize_path(path)
        return path
    elif os.path.isdir(path):
        images_path = []
        for root, _, filenames in os.walk(path):
            for filename in filenames:
                if filename.lower().endswith(('.jpg', '.bmp', '.png')):
                    img_path = os.path.join(root, filename)
                    images_path.append(img_path)
        return images_path
    elif os.path.isfile(path) and path.lower().endswith(('.jpg', '.bmp', '.png')):
        return [path]
    else:
        raise ValueError("The provided path is neither a directory nor a valid image file.")


def preprocess(img_path, crop_size=None, callback=None):
    """
    Preprocesses images from a given folder path by cropping and converting to the specified color format.

    Args:
        img_path (str): Path to the folder containing the images.
        crop_size (int, optional): Size of the cropped square. If None, the crop size will vary based on the image size.
        callback (function, optional): Function to call to update the progress. If None, the function will use print_progress_bar.
    Returns:
        images_arr (list): List of preprocessed image arrays.
        paths (list): List of paths to the processed images.
    """
    if crop_size is None:
        vary_crop = True
    else:
        vary_crop = False

    images_arr = []
    start_time = time.time()
    i = 0

    paths = load_images(img_path)
    n = len(paths)

    for path in paths:
        i += 1

        img = Image.open(path).convert('RGB')

        if vary_crop:
            crop_size = min(img.size)
        cropped = img.crop((0, 0, crop_size, crop_size))
        img_arr = np.array(cropped)

        if img_arr.shape[2] == 4:
            img_arr = img_arr[:, :, :-1]
        elif img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=-1)

        images_arr.append(img_arr)

        # Check if callback is provided
        if callback:
            callback('Preprocessed', i, n, start_time)
        else:
            print_progress_bar('Preprocessed', i, n, start_time=start_time)

    # Notify the end of the task
    if callback:
        callback(f'\nPreprocessing done. Please wait for entropy calculation to start.', n, n)
    else:
        print(f'\nPreprocessing done. Please wait for entropy calculation to start.')

    return images_arr, paths


def get_google_map_image(location, zoom_level, width=500, height=500, save=False):
    """
    Retrieves a satellite image from Google Maps for the specified location, zoom level, and dimensions.

    Args:
        location (str): Name of location, or its coordinates.
        zoom_level (int): Zoom level for the map image.
        width (int, optional): Width of the image in pixels. Default is 500.
        height (int, optional): Height of the image in pixels. Default is 500.
        save (bool, optional): Whether to save the image to a file. Default is False.

    Returns:
        image (np.ndarray): Array representing the retrieved image.

    Raises:
        Exception: If there is an error retrieving the image.

    Docs:
        https://developers.google.com/maps/documentation/maps-static/start
    """
    url = "https://maps.googleapis.com/maps/api/staticmap"
    # api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    api_key = 'AIzaSyAnFhz63LUhs9BGZfU_MW5EATc-s9r7epQ'
    cut = 50
    params = {
        "center": location,
        "zoom": zoom_level,
        "size": f"{width}x{height + cut}",
        "maptype": "satellite",
        "key": api_key,
        "scale": 2
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image = image.crop((0, 0, image.width, image.height - cut))

        if save:
            if not os.path.exists(save):
                os.makedirs(save)
            timestamp = time.strftime('%Y%m%d%H%M%S')
            filename = f"{save}/map_image_{location.replace(',', '_')}_{timestamp}.png"
            image.save(filename)
            print(f'Saved picture location: {location}, Zoom level: {zoom_level}')

        image = np.array(image)
        return image
    else:
        raise Exception(f"Error retrieving image: {response.text}")


def random_satellite_img(file_path, zoom_level=14, n_pics=10, save_path=f"../datasets/satellite"):
    """
    Generates random satellite images based on given rectangular coordinates.

    Parameters:
    - file_path (str): The path to the JSON file containing the rectangular coordinates.
    - zoom_level (int, optional): The zoom level for the satellite images. Default is 14.
    - n_pics (int, optional): The number of pictures to generate for each rectangle. Default is 10.
    - save_path (str, optional): The path where the generated images will be saved. Default is "../datasets/satellite".

    Returns:
    None
    """
    with open(file_path, 'r') as f:
        rects = json.load(f)

    n = n_pics * len(rects)
    j = 0
    for rect in rects:
        for i in range(n_pics):
            j += 1
            coo = random_point_in_rectangle(rect)
            img = get_google_map_image(coo, zoom_level, save=save_path)
            print(f'Got picture: {j}/{n}')
            # print(f"Image location: {coo}.")
            # print(f"Image entropy: {calc_ent(img, 'naive')}.")


def parse_coordinate(coordinate_str):
    """
    Parse a coordinate string in the format "valueN/S, valueE/W" and convert to numerical values.

    :param coordinate_str: String containing latitude and longitude in the format "valueN/S, valueE/W".
    :return: Tuple of (latitude, longitude) as floating-point numbers. Latitude is in the range -90 to 90,
             and longitude is in the range -180 to 180.
    """
    if "N" in coordinate_str or "S" in coordinate_str:
        if ", " in coordinate_str:
            latitude, longitude = coordinate_str.split(", ")
        else:
            latitude, longitude = coordinate_str.split(",")
        lat_value, lat_dir = float(latitude[:-1]), latitude[-1]
        lon_value, lon_dir = float(longitude[:-1]), longitude[-1]

        if lat_dir == 'S':
            lat_value = -lat_value
        if lon_dir == 'W':
            lon_value = -lon_value

        return lat_value, lon_value
    else:
        return tuple(map(float, coordinate_str.split(", ")))


def random_point_in_rectangle(coo):
    """
    Randomly select a point within a rectangle defined by coordinates in coo.

    :param coo: List of two strings, each containing "latitude, longitude" for the corners of the rectangle.
    :return: String of "latitude, longitude" for the randomly selected point.
    """
    top_left = parse_coordinate(coo[0])
    bottom_right = parse_coordinate(coo[1])

    # Randomly select a latitude between the top-left and bottom-right latitudes
    random_latitude = random.uniform(bottom_right[0], top_left[0])

    # Randomly select a longitude between the top-left and bottom-right longitudes
    random_longitude = random.uniform(top_left[1], bottom_right[1])

    # Format the coordinates as a string
    coordinates_str = "{:.4f}, {:.4f}".format(random_latitude, random_longitude)

    return coordinates_str


def ent_for_img(path, methods, ent_norm=None):
    img_arr = np.array(Image.open(path))
    for ent in methods:
        s = calc_ent(img_arr, ent, ent_norm)
        print(f'Entropy: {s},  Method: {ent}')


def get_ent_norm(method=None):
    with open('data/ent_norm.json', 'r') as file:
        ent_norm = json.load(file)
    if isinstance(method, str):
        method = [method]

    if method is not None:
        for met in method:
            if met not in ent_norm:
                fixed_noise = np.array(Image.open('../datasets/fixed_noise.bmp'))
                ent_norm[met] = calc_ent(fixed_noise, met)
                with open('data/ent_norm.json', 'w') as file:
                    json.dump(ent_norm, file)

    return ent_norm


def norm_weights(methods):
    pass
