import numpy as np
from scipy.stats import entropy
from PIL import Image
import os


def calc_ent(img_arr, method):
    img_entropy = 0
    match method:
        case 'hist':
            bins_ = 256
            hists, bins = np.histogram(img_arr.ravel(), bins=bins_, range=(0, bins_))
            hists_done = hists / hists.sum()
            img_entropy = entropy(hists_done)
    return img_entropy


def save_img(path, arr):
    Image.fromarray(arr).convert('RGB').save(path)


def load_images(path):
    imgs_path = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        imgs_path.append(img_path)
    return imgs_path


def preprocess(path, crop_size=None):
    img = Image.open(path)
    if crop_size is None:
        crop_size = min(img.size)
    cropped = img.crop((0, 0, crop_size, crop_size))
    img_arr = np.array(cropped)
    return img_arr


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
    noise_level = int(128 * noise_level)
    height, width = im_arr.shape[:2]
    noise_arr = np.zeros((height, width), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            noise_arr[i][j] = np.random.uniform(-noise_level, noise_level)
    noise_arr = (im_arr + noise_arr) % 255
    return noise_arr


def custom_permute(matrix, permutation_matrix=None):
    size = matrix.shape[0] * matrix.shape[1]
    """
        Permutes the elements of a matrix based on a permutation matrix.

        Args:
            matrix (numpy.ndarray): The matrix to be permuted.
            permutation_matrix (numpy.ndarray, optional): A matrix specifying the new positions
                of elements after permutation. If not provided, a random permutation will be used.

        Returns:
            numpy.ndarray: The permuted matrix.
    """
    if permutation_matrix is None:
        permutation_matrix = np.random.permutation(size) + 1
    flat_matrix = matrix.flatten()
    flat_permutation = permutation_matrix.flatten()
    permuted_matrix = np.array([flat_matrix[flat_permutation[i] - 1] for i in range(size)])
    reshaped_matrix = permuted_matrix.reshape(matrix.shape)

    return reshaped_matrix
