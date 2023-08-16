import os
import platform
import subprocess
from io import BytesIO

import requests

from entropy import *
from img_utils import *


def calc_ent(img_arr, method):
    match method:
        case 'hist':
            return histogram(img_arr, 'rgb')
        case 'hist_greyscale':
            return histogram(img_arr, 'greyscale')
        case 'naive':
            return entropy(img_arr)
        case 'dft':
            return calc_dft(img_arr)
        case 'dwt':  # Haven't finish
            pass
        case 'laplace':
            return laplace_ent(img_arr)
        case 'joint_red_green':
            return calculate_joint_entropy_red_green(img_arr)
        case 'joint_all':
            return calculate_joint_RGB_entropy(img_arr)
        case 'lbp':
            return calculate_texture_entropy(img_arr)
        case 'lbp_gabor':
            return calculate_texture_gabor_entropy(img_arr)
        case 'cube':
            return calculate_rgb_color_cube_entropy(img_arr)
        case 'GLCM':
            return calculate_GLCM_entropy(img_arr)
        case 'RGBCM_each_channel':
            return calculate_RGBCM_entropy(img_arr, scheme='each_channel')
        case 'RGBCM_to_gray':
            return calculate_RGBCM_entropy(img_arr, scheme='to_gray')
        case _:
            print('No entropy method matched!!')
            return


def save_img(folder_path, img):
    if isinstance(img, np.ndarray):
        img = [img]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    i = 0
    for arr in img:
        path = os.path.join(folder_path, f'i={i}_s={arr[1]:.3f}.bmp')
        Image.fromarray(arr[0]).save(path)
        i += 1

    print(f'Sorted images saved to: {os.path.abspath(folder_path)}')
    if platform.system() == 'Windows':
        os.startfile(os.path.join(os.getcwd(), folder_path))
    elif platform.system() == 'Darwin':
        subprocess.run(['open', os.path.join(os.getcwd(), folder_path)])
    else:
        print("Unsupported OS")


def load_images(path):
    images_path = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        images_path.append(img_path)
    return images_path


def preprocess(folder_path, crop_size=None, colors='rgb'):
    if crop_size is None:
        vary_crop = True
    else:
        vary_crop = False

    paths = load_images(folder_path)
    n = len(paths)
    i = 0

    images_arr = []
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
        print(f'\rPreprocessed: {print_progress_bar(i, n)} {i}/{n} images.', end='', flush=True)

    print(f'\nPreprocessing done.')

    return images_arr


def label_ent(images, method):
    img_ent = []
    n = len(images)
    i = 0
    for img in images:
        i += 1
        img_ent.append([img, calc_ent(img, method)])
        print(f'\rEntropy calculated: {print_progress_bar(i, n)} {i}/{n} images.', end='', flush=True)
    print('\nEntropy calculation done.')

    return img_ent


def normalize_path(path_str):
    # Split by both UNIX and Windows separators
    parts = path_str.replace('\\', '/').split('/')
    # Join with the appropriate OS separator
    return os.path.join(*parts)


def print_progress_bar(iteration, total, length=50):
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    return "█" * filled_length + '-' * (length - filled_length)


def get_google_map_image(latitude, longitude, zoom_level, width=500, height=500, save=False):
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
