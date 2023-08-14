import math

import matplotlib.pyplot as plt

from scripts.transforms import *


def entropy(arr):
    # Check the rank of the array
    rank = arr.ndim

    # Handle 1D and 2D arrays
    if rank == 1 or rank == 2:
        total_sum = np.sum(arr)
        normalize_arr = arr / total_sum
        ent = 0
        for data in normalize_arr.flatten():
            ent -= data * math.log(data + np.finfo(float).eps, 2)
        return ent

    # Handle 3D arrays
    elif rank == 3:
        total_entropy = 0
        for slice_2d in arr:
            total_entropy += entropy(slice_2d)  # Recursively call the function for each 2D slice
        return total_entropy

    else:
        raise ValueError("Array must be 1D, 2D, or 3D")


def histogram(img_arr, color='rgb'):
    match color:
        case 'rgb':
            bins_ = 256 ** 3
            flattened_img_arr = (img_arr[:, :, 0] << 16) + (img_arr[:, :, 1] << 8) + img_arr[:, :, 2]
            hist, _ = np.histogram(flattened_img_arr, bins=bins_, range=(0, bins_ - 1))
            return entropy(hist)
        case 'greyscale':
            bins_ = 256
            hist, _ = np.histogram(img_arr.ravel(), bins=bins_, range=(0, bins_))
            return entropy(hist)


def calc_dft(image, visualize=False):
    # Compute the 2D Fourier Transform of the image
    f_transform = dft(image)
    if visualize:
        f_transform_centered = np.fft.fftshift(f_transform)
        # Compute magnitude spectrum (for visualization purposes)
        magnitude_spectrum = np.log(np.abs(f_transform_centered) + 1)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum')
        plt.colorbar()
        plt.show()
    return entropy(f_transform)


def laplace_ent(image):
    return entropy(laplacian(image))
