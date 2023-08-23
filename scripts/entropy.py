import math

from scipy.signal import convolve2d
from skimage.color import rgb2gray
from skimage.feature import graycomatrix
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
from skimage.segmentation import slic

from transforms import *


def label_ent(images, method, sort=True, ent_norm=None, colors='rgb', color_weight=None):
    """
    Calculates entropy for a list of images using the specified method and optionally sorts them by entropy.

    Args:
        images (list): List of image arrays.
        method (str): Name of method to use for entropy calculation.
        sort (bool, optional): Whether to sort the images by entropy. Default is True.
        ent_norm (dict, optional): Entropy normalization dictionary.
        colors (str, optional): Which color channels to use for entropy calculation.

    Returns:
        img_ent (list): List of tuples containing the image array and its corresponding entropy.
    """
    img_ent = []
    n = len(images)
    i = 0
    start_time = time.time()

    for img in images:
        i += 1

        img_ent.append([img, calc_ent(change_channels(img, colors), method, ent_norm=ent_norm, color_weight=color_weight)])
        print_progress_bar('Entropy calculated', i, n, start_time=start_time)
    print('\nEntropy calculation done.')

    if sort:
        img_ent = sorted(img_ent, key=lambda x: x[1])
    return img_ent


def calc_ent(img_arr, method, ent_norm=None, color_weight=None):
    """
    Calculates the entropy of an image array using the specified method.

    Args:
        img_arr (np.ndarray): Image array for which to calculate entropy.
        method (str): Method to use for entropy calculation. Supported methods include:
            - 'hist': Histogram-based entropy for RGB images.
            - 'hist_greyscale': Histogram-based entropy for greyscale images.
            - 'naive': Naive entropy calculation.
            - 'dft': Entropy calculation using Discrete Fourier Transform.
            - 'laplace': Laplace entropy calculation.
            - 'joint_red_green': Joint entropy calculation for red and green channels.
            - 'joint_all': Joint entropy calculation for RGB channels.
            - 'lbp': Local Binary Pattern-based texture entropy.
            - 'lbp_gabor': Texture entropy using Local Binary Pattern and Gabor filter.
            - 'adapt': Adaptive entropy estimation.
            - 'GLCM': Entropy calculation using Gray-Level Co-occurrence Matrix.
            - 'RGBCM_each_channel': Entropy calculation using Red-Green-Blue Co-occurrence Matrix for each channel.
            - 'RGBCM_to_gray': Entropy calculation using Red-Green-Blue Co-occurrence Matrix converted to grayscale.
        ent_norm (dict, optional): Normalization dictionary to normalize the entropy based on a fixed image.

    Returns:
        float: Calculated entropy value, or None if the method is not recognized.

    Note:
        Some methods may require specific functions to be defined elsewhere in the code.
    """

    match method:
        case 'hist':
            ent = histogram(img_arr)
        case 'naive':
            ent = entropy(img_arr)
        case 'dft':
            ent = calculate_dft_entropy(img_arr)
        case 'dwt':
            ent = calculate_dwt_entropy(img_arr)
        case 'laplace':
            ent = laplace_ent(img_arr)
        case 'joint_red_green':
            ent = calculate_joint_entropy_red_green(img_arr)
        case 'joint_all':
            ent = calculate_joint_RGB_entropy(img_arr)
        case 'lbp':
            ent = calculate_texture_entropy(img_arr)
        case 'lbp_gabor':
            ent = calculate_texture_gabor_entropy(img_arr)
        case 'adapt':
            ent = adaptive_entropy_estimation(img_arr)
        case 'GLCM':
            ent = calculate_GLCM_entropy(img_arr)
        case 'RGBCM_each_channel':
            ent = calculate_RGBCM_entropy(img_arr, scheme='each_channel')
        case 'RGBCM_to_gray':
            ent = calculate_RGBCM_entropy(img_arr, scheme='to_gray')
        case _:
            print('No entropy method matched!!')
            ent = None

    if ent_norm is not None:
        ent /= ent_norm[method]

    return ent


def entropy(arr, color_weight=(0.2989, 0.5870, 0.1140)):
    arr = np.abs(arr)

    if arr.ndim == 1 or arr.ndim == 2:
        total_sum = np.sum(arr)
        if total_sum == 0:
            return 0
        normalize_arr = arr / total_sum
        return -np.sum(normalize_arr * np.log2(normalize_arr + np.finfo(float).eps))

    elif arr.ndim == 3:
        if arr.shape[-1] != 3:  # Check if the last dimension has 3 channels (RGB)
            raise ValueError("entropy function: 3D array must represent an RGB image with three channels")

        weighted_arr = np.dot(arr, color_weight)
        return entropy(weighted_arr)
    else:
        raise ValueError("Array must be 1D, 2D, or 3D")

def histogram(img_arr):
    if img_arr.ndim == 3:  # Check if the tensor is RGB (rank 3)
        bins_ = 256 ** 3
        flattened_img_arr = (img_arr[:, :, 0] << 16) + (img_arr[:, :, 1] << 8) + img_arr[:, :, 2]
        hist, _ = np.histogram(flattened_img_arr, bins=bins_, range=(0, bins_ - 1))
        return entropy(hist)
    elif img_arr.ndim == 2:  # Check if the tensor is grayscale (rank 2)
        bins_ = 256
        hist, _ = np.histogram(img_arr.ravel(), bins=bins_, range=(0, bins_))
        return entropy(hist)
    else:
        raise ValueError("Invalid tensor rank. Supported ranks are 2 (greyscale) and 3 (RGB).")


def calculate_GLCM_entropy(image):
    # Check if the image is 3D (e.g., RGB) and convert to grayscale if necessary
    if len(image.shape) == 3:
        image = rgb2gray(image)

    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    levels = 256

    gray_image = (image * 255).astype(np.uint8)
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=levels, symmetric=False, normed=True)

    total_entropy = 0
    for angle_idx in range(len(angles)):
        matrix = glcm[:, :, 0, angle_idx]
        matrix = matrix / np.sum(matrix)
        entropy_ = -np.sum(matrix * np.log2(matrix + np.finfo(float).eps))
        total_entropy += entropy_

    return total_entropy


def calculate_RGBCM_entropy(image, scheme='each_channel'):
    distances = [1]  # Distance between pixels for co-occurrence
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Angles for co-occurrence (in radians)
    levels = 256  # Number of intensity levels in the image

    total_entropy = 0

    def calculate_entropy(matrix_):
        matrix_ = matrix_ / np.sum(matrix_)  # Normalize matrix to probabilities
        entropy_ = -np.sum(matrix_ * np.log2(matrix_ + np.finfo(float).eps))  # Avoid log(0)
        return entropy_

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
                entropy_ = calculate_entropy(matrix)
                channel_entropy += entropy_

            total_entropy += channel_entropy

    elif scheme == 'to_gray':
        # Convert the RGB image to grayscale and calculate entropy
        gray_image = rgb2gray(image)
        gray_image = (gray_image * 255).astype(np.uint8)
        glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=levels, symmetric=False, normed=True)

        for angle_idx in range(len(angles)):
            matrix_ = glcm[:, :, 0, angle_idx]
            entropy_ = calculate_entropy(matrix_)
            total_entropy += entropy_

    return total_entropy


def calculate_dft_entropy(image):
    dft_img = dft(image)
    return entropy(dft_img)


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
    # Ensure img_arr is a numpy array
    img_arr = np.asarray(img_arr)

    # Convert the image to grayscale if it's a color image
    if img_arr.ndim == 3 and img_arr.shape[-1] == 3:  # Check for RGB image
        gray_image = rgb2gray(img_arr)
    elif img_arr.ndim == 2 or (img_arr.ndim == 3 and img_arr.shape[-1] == 1):  # Grayscale or single-channel image
        gray_image = img_arr.squeeze()
    else:
        raise ValueError("Input image should be either grayscale or RGB.")

    # Normalize the grayscale image if it isn't already
    if gray_image.max() > 1:
        gray_image = gray_image / 255.0

    # Convert to 8-bit integer type
    gray_image = (gray_image * 255).astype(np.uint8)

    # Apply Local Binary Pattern (LBP) to extract texture features
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')

    # Calculate histogram of LBP values
    n_bins = int(n_points * (n_points - 1) / 2) + 2  # for 'uniform' method with n_points=8, this is 59 + 1 = 60
    hist, _ = np.histogram(lbp_image, bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + np.finfo(float).eps)

    # Calculate texture entropy
    texture_entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))

    return texture_entropy


def calculate_texture_gabor_entropy(img_arr):
    # Ensure img_arr is a numpy array
    img_arr = np.asarray(img_arr)

    # Convert the image to grayscale if it's a color image
    if img_arr.ndim == 3 and img_arr.shape[-1] in [3, 4]:  # Check for RGB or RGBA image
        gray_image = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]
    elif img_arr.ndim == 2 or (img_arr.ndim == 3 and img_arr.shape[-1] == 1):  # Grayscale or single-channel image
        gray_image = img_arr.squeeze()
    else:
        raise ValueError("Input image should be either grayscale or RGB.")

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
    hist, _ = np.histogram(gabor_response, bins=256, density=True)
    entropy_value = entropy(hist + np.finfo(float).eps)  # Add small value to avoid log(0)

    return entropy_value


def adaptive_entropy_estimation(img_arr, num_segments=100):
    # Convert the image to grayscale
    if len(img_arr.shape) == 3:
        gray_image = rgb2gray(img_arr)
    else:
        gray_image = img_arr

    # Segment the image using SLIC
    segments = slic(img_arr, n_segments=num_segments, compactness=10, sigma=1)

    segment_entropies = []
    unique_segments = np.unique(segments)

    for segment_idx in unique_segments:
        segment_mask = (segments == segment_idx)
        segment_region = gray_image[segment_mask]
        segment_entropy = shannon_entropy(segment_region)
        segment_entropies.append(segment_entropy)

    total_entropy = np.sum(segment_entropies)
    adaptive_entropy = total_entropy / len(unique_segments)

    return adaptive_entropy


def laplace_ent(image):
    return entropy(laplacian(image))


def calculate_dwt_entropy(image, wavelet='db1', level=1):
    w_transform_arr = dwt(image, wavelet, level)
    return entropy(w_transform_arr)
