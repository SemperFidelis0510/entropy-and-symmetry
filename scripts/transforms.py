import colorsys

import numpy as np
import pywt

from utils import *


def dft(image):
    """
    Computes the Discrete Fourier Transform (DFT) of a 1D, 2D, or 3D array (image).

    Args:
        image (np.ndarray): Input array (image) for which to compute the DFT. Can be 1D, 2D, or 3D.

    Returns:
        np.ndarray: Absolute values of the Fourier Transform, with the same shape as the input array.

    Raises:
        ValueError: If the input array is not 1D, 2D, or 3D.

    Note:
        For 3D arrays, the DFT is computed for each 2D slice along the third dimension.
    """
    rank = image.ndim

    # Handle 1D arrays
    if rank == 1:
        f_transform = np.fft.fft(image)
        return np.abs(f_transform)

    # Handle 2D arrays
    elif rank == 2:
        f_transform = np.fft.fft2(image)
        return np.abs(f_transform)

    # Handle 3D arrays
    elif rank == 3:
        result = np.empty_like(image, dtype=np.float64)
        for i in range(image.shape[2]):
            slice_2d = image[:, :, i]
            f_transform = np.fft.fft2(slice_2d)
            result[:, :, i] = np.abs(f_transform)
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


def laplacian(x):
    rank = x.ndim

    # Handle 1D arrays
    if rank == 1:
        result = np.zeros_like(x)
        for i in range(len(x)):
            result[i] = 2 * x[i]
            if i > 0:
                result[i] -= x[i - 1]
            if i < len(x) - 1:
                result[i] -= x[i + 1]
        return result

    # Handle 2D arrays
    elif rank == 2:
        rows, cols = x.shape
        result = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                result[i, j] = 4 * x[i, j]
                if i > 0:
                    result[i, j] -= x[i - 1, j]
                if i < rows - 1:
                    result[i, j] -= x[i + 1, j]
                if j > 0:
                    result[i, j] -= x[i, j - 1]
                if j < cols - 1:
                    result[i, j] -= x[i, j + 1]
        return result

    # Handle 3D arrays
    elif rank == 3:
        result = np.empty_like(x, dtype=np.float64)
        for i in range(x.shape[2]):
            slice_2d = x[:, :, i]
            result[:, :, i] = laplacian(slice_2d)  # Recursively call the function for each 2D slice
        return result

    else:
        raise ValueError("Array must be 1D, 2D, or 3D")


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


def rgb_to_hsb_image(rgb_image):
    # Normalize RGB values to [0, 1]
    rgb_image = rgb_image / 255.0

    # Initialize an empty array for HSB values
    hsb_image = np.zeros_like(rgb_image, dtype=float)

    # Iterate through the pixels and convert RGB to HSB
    for i in range(rgb_image.shape[0]):
        for j in range(rgb_image.shape[1]):
            r, g, b = rgb_image[i, j]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hsb_image[i, j] = (h * 360, s * 100, v * 100)  # Convert to degrees and percentage

    return hsb_image
