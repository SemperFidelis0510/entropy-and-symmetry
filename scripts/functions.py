import math
import os

import numpy as np
from PIL import Image
from scipy.stats import entropy
import matplotlib.pyplot as plt

def calc_ent(img_arr, method):
    img_entropy = 0
    match method:
        case 'hist_greyscale':
            bins_ = 256
            hists, bins = np.histogram(img_arr.ravel(), bins=bins_, range=(0, bins_))
            hists_done = hists / hists.sum()
            img_entropy = entropy(hists_done)
        case 'naive':
            for i in range(3):
                img_entropy += S(img_arr[:, :, i])
        case 'hist':
            bins_ = 256 ** 3
            flattened_img_arr = (img_arr[:, :, 0] << 16) + (img_arr[:, :, 1] << 8) + img_arr[:, :, 2]
            hist, _ = np.histogram(flattened_img_arr, bins=bins_, range=(0, bins_ - 1))
            hist_done = hist / np.sum(hist)
            img_entropy = -np.sum(hist_done * np.log(hist_done + np.finfo(float).eps))
    return img_entropy


def save_img(folder_path, img):
    if isinstance(img, np.ndarray):
        img = [img]
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    i = 0
    for arr in img:
        path = os.path.join(folder_path, f'{i}_s={arr[1]:.3f}.bmp')
        Image.fromarray(arr[0]).save(path)
        i += 1

    print(f'All pictures saved to folder {folder_path}.')


def load_images(path):
    imgs_path = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        imgs_path.append(img_path)
    return imgs_path


def preprocess(folder_path, crop_size=None, colors='rgb'):
    if crop_size is None:
        vary_crop = True
    else:
        vary_crop = False

    paths = load_images(folder_path)
    n = len(paths)
    i = 0

    imgs_arr = []
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

        imgs_arr.append(img_arr)
        print(f'Processed {i}/{n} images.')

    print(f'All the images from "{folder_path}" were loaded, and preprocessed.')

    return imgs_arr


def label_ent(imgs, method):
    img_ent = []
    n = len(imgs)
    i = 0
    for img in imgs:
        i += 1
        img_ent.append([img, calc_ent(img, method)])
        print(f'Entropy calculated for {i}/{n} images.')

    return img_ent


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


def S(arr_2d):
    total_sum = np.sum(arr_2d)
    normalize_arr = arr_2d / total_sum
    height, width = normalize_arr.shape[:2]
    print(normalize_arr.shape)
    ent = 0
    for x in range(height):
        for y in range(width):
            data = normalize_arr[x, y]
            ent -= data * math.log(data + np.finfo(float).eps)
    return ent


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

def compute_dft(image, visualize=False):
    # Compute the 2D Fourier Transform of the image
    f_transform = np.fft.fft2(image)
    # Shift zero frequency component to the center
    f_transform_centered = np.fft.fftshift(f_transform)
    if visualize == True:
        # Compute magnitude spectrum (for visualization purposes)
        magnitude_spectrum = np.log(np.abs(f_transform_centered) + 1)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum')
        plt.colorbar()
        plt.show()
    return f_transform