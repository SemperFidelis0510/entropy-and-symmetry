import math

import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import graycomatrix
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from scipy.signal import convolve2d
from transforms import *


def entropy(arr):
    # Check the rank of the array
    rank = arr.ndim
    arr = np.abs(arr)

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


def calculate_GLCM_entropy(image):
    distances = [1]  # Distance between pixels for co-occurrence
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Angles for co-occurrence (in radians)
    levels = 256  # Number of intensity levels in the image

    gray_image = (image * 255).astype(np.uint8)
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=levels, symmetric=False, normed=True)

    total_entropy = 0
    for angle_idx in range(len(angles)):
        matrix = glcm[:, :, 0, angle_idx]
        matrix = matrix / np.sum(matrix)  # Normalize matrix to probabilities
        entropy = -np.sum(matrix * np.log2(matrix + np.finfo(float).eps))
        total_entropy += entropy

    return total_entropy


def calculate_RGBCM_entropy(image, scheme='each_channel'):
    distances = [1]  # Distance between pixels for co-occurrence
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Angles for co-occurrence (in radians)
    levels = 256  # Number of intensity levels in the image

    total_entropy = 0

    def calculate_entropy(matrix):
        matrix = matrix / np.sum(matrix)  # Normalize matrix to probabilities
        entropy = -np.sum(matrix * np.log2(matrix + np.finfo(float).eps))  # Avoid log(0)
        return entropy

    if scheme == 'each_channel':
        # Calculate entropy for each color channel separately
        for channel in range(3):  # Iterate over R, G, and B channels
            channel_image = image[:, :, channel]
            gray_image = (channel_image * 255).astype(np.uint8)
            glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=levels, symmetric=False,
                                normed=True)

            channel_entropy = 0
            for angle_idx in range(len(angles)):
                matrix = glcm[:, :, 0, angle_idx]
                entropy = calculate_entropy(matrix)
                channel_entropy += entropy

            total_entropy += channel_entropy

    elif scheme == 'to_gray':
        # Convert the RGB image to grayscale and calculate entropy
        gray_image = color.rgb2gray(image)
        gray_image = (gray_image * 255).astype(np.uint8)
        glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=levels, symmetric=False, normed=True)

        for angle_idx in range(len(angles)):
            matrix = glcm[:, :, 0, angle_idx]
            entropy = calculate_entropy(matrix)
            total_entropy += entropy

    return total_entropy


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


def calculate_joint_entropy_red_green(img_arr):
    red_channel = img_arr[:, :, 0]
    green_channel = img_arr[:, :, 1]

    joint_histogram, _, _ = np.histogram2d(red_channel.ravel(), green_channel.ravel(), bins=256)

    joint_probabilities = joint_histogram / np.sum(joint_histogram)

    joint_entropy = -np.sum(joint_probabilities * np.log2(joint_probabilities + np.finfo(float).eps))
    return joint_entropy


def calculate_joint_RGB_entropy(rgb_image):
    # Calculate histograms for each color channel
    hist_r, _ = np.histogram(rgb_image[:, :, 0], bins=256, range=(0, 256))
    hist_g, _ = np.histogram(rgb_image[:, :, 1], bins=256, range=(0, 256))
    hist_b, _ = np.histogram(rgb_image[:, :, 2], bins=256, range=(0, 256))

    # Normalize histograms to obtain probability distributions
    p_r = hist_r / np.sum(hist_r)
    p_g = hist_g / np.sum(hist_g)
    p_b = hist_b / np.sum(hist_b)

    # Calculate joint entropy using the definition of joint entropy
    joint_entropy = 0
    for i in range(256):
        for j in range(256):
            for k in range(256):
                if p_r[i] > 0 and p_g[j] > 0 and p_b[k] > 0:
                    joint_prob = p_r[i] * p_g[j] * p_b[k]
                    joint_entropy -= joint_prob * np.log2(joint_prob)

    return joint_entropy


def calculate_texture_entropy(img_arr):
    # Convert the image to grayscale
    if len(img_arr.shape) == 3:
        gray_image = rgb2gray(img_arr)
    else:
        gray_image = img_arr
    # Apply Local Binary Pattern (LBP) to extract texture features
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')

    # Calculate histogram of LBP values
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 2 ** n_points))
    hist = hist.astype("float")
    hist /= (hist.sum() + np.finfo(float).eps)

    texture_entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))

    return texture_entropy


def calculate_texture_gabor_entropy(img_arr):
    # Convert the image to grayscale
    gray_image = np.mean(img_arr, axis=2)

    # Define parameters for Gabor filter
    wavelength = 5.0
    orientation = np.pi / 4
    frequency = 1 / wavelength
    sigma = 1.0

    # Create Gabor filter
    x, y = np.meshgrid(np.arange(-15, 16), np.arange(-15, 16))
    gabor_real = np.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2)) * np.cos(
        2 * np.pi * frequency * (x * np.cos(orientation) + y * np.sin(orientation)))

    # Apply Gabor filter
    gabor_response = convolve2d(gray_image, gabor_real, mode='same', boundary='wrap')

    # Calculate image entropy
    flat_gabor_response = gabor_response.flatten()
    hist, _ = np.histogram(flat_gabor_response, bins=256, range=(-255, 255), density=True)
    entropy_value = entropy(hist)

    return entropy_value


def laplace_ent(image):
    return entropy(laplacian(image))
