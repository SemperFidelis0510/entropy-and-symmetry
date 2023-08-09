import numpy as np
from PIL import Image
import os

def import_imgs(path):
    """
    This function will load all image files specified in path
    and store them as array representation into a list.
    Args:
        path: which contains source images
    Returns:
        A list of image arrays.
    """
    imgs = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        imgs.append(Image.open(img_path))
    return imgs

def preprocess_imgs(imgs, crop_frame=(0,0,0,0)):
    img_arr = []
    for img in imgs:
        cropped = crop_img(img, crop_frame)
        converted = img_to_arr(cropped)
        img_arr.append(converted)
    return img_arr

def img_to_arr(img):
    return np.array(img)

def crop_img(img, frame = (0,0,0,0)):
    """
    This function will crop image according to the frame, if
    the frame is not specified, the common maximal square frame will be applied
    Args:
        img_path: Path to file
        frame: A 4-tuple specified left, top, right, bottom position of the frame
    Returns:
        The cropped image represented by array
    """
    if frame == (0,0,0,0):
        frame = cal_frame()
    return img.crop(frame)


def cal_frame(imgs):
    """
    This function will calculate common maximal square frame among all images
    Args:
        imgs: List of images represented by array
    Returns:
        Common maximal square frame
    """

def uniform_noise(im_arr, noise_level):
    """
    This function will add uniform noise to image.
    Args:
        im_arr: image represented by an array
        noise_level: From 0 to 1, 0 means original image, 1 means complete noise
    Returns:
        A new image represented by an array
    """
    if noise_level < 0 or noise_level > 1:
        print('error: noise function receive invalid parameter: noise_level should be in [0, 1]')
        return
    height, width = im_arr.shape[:2]
    noise_arr = np.zeros((height,width), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            noise_arr[i][j] = np.random.uniform(0,noise_level)
    noise_arr = im_arr/255 + noise_arr
    noise_arr = np.clip(noise_arr, 0, 1)
    return noise_arr