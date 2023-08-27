import colorsys

import numpy as np
import pywt
from skimage.color import rgb2gray
from scipy.ndimage import convolve

from utils import *


def dft(image):
    rank = image.ndim

    if rank == 1:
        return np.abs(np.fft.fft(image))
    elif rank == 2:
        return np.abs(np.fft.fft2(image))
    elif rank == 3:
        result = np.empty_like(image, dtype=np.float64)
        for i in range(image.shape[2]):
            result[:, :, i] = np.abs(np.fft.fft2(image[:, :, i]))
        return result
    else:
        raise ValueError("Array must be 1D, 2D, or 3D")


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
    noise_arr = np.random.uniform(-noise_level, noise_level, im_arr.shape)
    noise_arr = (im_arr + noise_arr) % 255
    return noise_arr.astype(im_arr.dtype)


def noise_by_increment(im_arr, num_images):
    """
    This function generates a list of arrays of noised-up pictures with rising values of noise.
    Args:
        im_arr: image represented by an array
        num_images: Number of images to generate in the list
    Returns:
        A list of noised images represented by arrays
    """
    noised_images = [im_arr.copy()]  # Start with the original clean image
    for i in range(1, num_images):
        noise_level = i / (num_images - 1) if num_images > 1 else 0
        noised_image = uniform_noise(im_arr, noise_level)
        noised_images.append(noised_image)
        print_progress_bar('Noised up images', i + 1, num_images)
    print('\nNosing up images done.')
    return noised_images


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


def laplacian(arr):
    rank = arr.ndim

    # Define the kernel based on the array's rank
    if rank == 1:
        kernel = np.array([-1, 2, -1])
    elif rank == 2:
        kernel = np.array([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]])
    elif rank == 3:
        kernel = np.zeros((3, 3, 3))
        kernel[1, 1, 1] = 6
        kernel[1, 1, 0] = kernel[1, 1, 2] = kernel[1, 0, 1] = kernel[1, 2, 1] = kernel[0, 1, 1] = kernel[2, 1, 1] = -1
    else:
        raise ValueError("Array must be 1D, 2D, or 3D")

    # Apply the convolution with the kernel
    result = convolve(arr, kernel, mode='constant', cval=0.0)

    return result


def dwt(image, wavelet='db1', level=None):
    rank = image.ndim

    # Handle 1D arrays
    if rank == 1:
        w_transform = pywt.wavedec(image, wavelet=wavelet, level=level)
        return np.abs(w_transform[level])

    # Handle 2D arrays
    elif rank == 2:
        w_transform = pywt.wavedec2(image, wavelet=wavelet, level=level)
        return np.abs(w_transform[level])

    # Handle 3D arrays
    elif rank == 3:
        result = [np.abs(pywt.wavedec2(image[:, :, i], wavelet=wavelet, level=level)[level]).flatten()
                  for i in range(image.shape[2])]
        return np.vstack(result)

    else:
        raise ValueError("Array must be 1D, 2D, or 3D")


def change_channels(img, channels):
    match channels:
        case 'rgb':
            img = img
        case 'hsb':
            # Normalize RGB values to [0, 1]
            img = img / 255.0

            # Initialize an empty array for HSB values
            hsb_image = np.zeros_like(img, dtype=float)

            # Iterate through the pixels and convert RGB to HSB
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    r, g, b = img[i, j]
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)
                    hsb_image[i, j] = (h * 360, s * 100, v * 100)  # Convert to degrees and percentage

            img = hsb_image

        case 'YCbCr':
            kR = 0.299
            kG = 0.587
            kB = 0.114
            R = img[..., 0]
            G = img[..., 1]
            B = img[..., 2]
            y = kR * R + kG * G + kB * B
            cb = (-kR / (2 * (1 - kB))) * R + (-kG / (2 * (1 - kB))) * G + 1 / 2 * B
            cr = 1 / 2 * R + (-kG / (2 * (1 - kR))) * G + (-kB / (2 * (1 - kR))) * B
            ycbcr_image_array = np.stack([y, cb, cr], axis=-1)
            img = ycbcr_image_array
        case 'greyscale':
            img = rgb2gray(img)

    return img
