from scipy.signal import convolve2d
from skimage.color import rgb2gray
from skimage.feature import graycomatrix
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
from skimage.segmentation import slic

from transforms import *


def linearCombine_ent(images, methods, method_weight=None, sort=True, ent_norm=None, colors='rgb',
                      color_weight=None,
                      callback=None):
    if isinstance(methods, str):
        methods = [methods]
    if method_weight is None:
        num_methods = len(methods)
        method_weight = [1 / num_methods] * num_methods
    ent_array = []

    for method in methods:
        ent_array.append(label_ent(images, method, sort, ent_norm=ent_norm, colors=colors, color_weight=color_weight,
                                   callback=callback))
        print(f'\nEntropy calculated by method: {method}')
    return np.dot(ent_array, method_weight)


def label_ent(images, methods, sort=True, ent_norm=None, colors='rgb', color_weight=None,
              callback=None):
    """
    Calculates entropy for a list of images using the specified method and optionally sorts them by entropy.

    Args:
        images (list): List of image arrays.
        methods (list/str): Name of method to use for entropy calculation.
        sort (bool, optional): Whether to sort the images by entropy. Default is True.
        ent_norm (dict, optional): Entropy normalization dictionary.
        colors (str, optional): Which color channels to use for entropy calculation.
        callback (function, optional): Callback function to use for progress bar.
        color_weight (tuple, optional): Decides the weights of contributions of each channel to the entropy.

    Returns:
        img_ent (list): List of tuples containing the image array and its corresponding entropy.
    """
    if isinstance(methods, str):
        methods = {methods: 1}
    elif isinstance(methods, list):
        num_methods = len(methods)
        methods = {met: 1 / num_methods for met in methods}

    img_ent = []
    n = len(images)
    i = 0
    start_time = time.time()

    for img in images:
        i += 1
        s_dict = {}
        for method in methods:
            s_dict[method] = calc_ent(change_channels(img, colors), method,
                                      ent_norm=ent_norm, color_weight=color_weight)

        c = complexity(s_dict, methods)
        img_ent.append([img, c])

        # Use the callback if provided, else use the default print_progress_bar
        if callback:
            callback('Entropy calculated', i, n, start_time=start_time)
        else:
            print_progress_bar('Entropy calculated', i, n, start_time=start_time)

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
        color_weight (tuple, optional): Decides the weights of contributions of each channel to the entropy.

    Returns:
        float: Calculated entropy value, or None if the method is not recognized.

    Note:
        Some methods may require specific functions to be defined elsewhere in the code.
    """
    transform_result = img_arr
    match method:
        case 'hist':
            transform_result = histogram(img_arr)
        case 'naive':
            pass
        case 'dft':
            transform_result = dft(img_arr)
        case 'dwt':
            transform_result = dwt(img_arr, wavelet='db1', level=1)
        case 'laplace':
            transform_result = laplace_ent(img_arr)
        case 'joint_red_green':
            transform_result = calculate_joint_entropy_red_green(img_arr)
        case 'joint_all':
            transform_result = calculate_joint_RGB_entropy(img_arr)
        case 'lbp':
            transform_result = calculate_texture_entropy(img_arr)
        case 'lbp_gabor':
            transform_result = calculate_texture_gabor_entropy(img_arr)
        case 'adapt':
            transform_result = adaptive_entropy_estimation(img_arr)
        case 'RGBCM':
            transform_result = calculate_CM_co_occurrence(img_arr)
        case _:
            raise ValueError(f"No entropy method matched for method '{method}'!!")

    if method != 'adapt':  # skip those already returned entropy
        ent = entropy(transform_result, color_weight=color_weight)
    else:
        ent = transform_result
    if ent_norm is not None:
        ent /= ent_norm[method]

    return ent


def entropy(arr, color_weight=None):
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
        if color_weight is not None:
            weighted_arr = np.dot(arr, color_weight)
        else:  # Default weighted
            weighted_arr = np.dot(arr, (0.2989, 0.5870, 0.1140))
        return entropy(weighted_arr)
    else:
        raise ValueError("Array must be 1D, 2D, or 3D")


def histogram(img_arr):
    if img_arr.ndim == 3:  # Check if the tensor is RGB (rank 3)
        img_arr = img_arr.astype(np.uint32)  # Convert to an integer type
        # Reduce color resolution by right-shifting
        reduced_img_arr = img_arr >> 4
        # Combine the reduced RGB values into a single integer
        flattened_img_arr = (reduced_img_arr[:, :, 0] << 12) + (reduced_img_arr[:, :, 1] << 6) + reduced_img_arr[:, :,
                                                                                                 2]
        # Create the histogram with fewer bins
        bins_ = 64 ** 3
        hist, _ = np.histogram(flattened_img_arr, bins=bins_, range=(0, bins_ - 1))
        return hist
    elif img_arr.ndim == 2:  # Check if the tensor is grayscale (rank 2)
        bins_ = 256
        hist, _ = np.histogram(img_arr.ravel(), bins=bins_, range=(0, bins_))
        return hist
    else:
        raise ValueError("Invalid tensor rank. Supported ranks are 2 (greyscale) and 3 (RGB).")


def calculate_CM_co_occurrence(image):
    """
    Calculate the color co-occurrence matrix for an RGB image at different angles.

    Parameters:
    - image: 3D NumPy array representing the RGB image.

    Returns:
    - 3D NumPy array representing the accumulated co-occurrence matrices for each color channel.
    """
    distances = [1]  # Distance between pixels for co-occurrence
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Angles for co-occurrence (in radians)
    levels = 256  # Number of intensity levels in the image

    # Initialize array for co-occurrence matrices
    co_occurrence_array = np.zeros((levels, levels, 3))

    for channel in range(3):  # Iterate over RGB channels
        channel_image = image[:, :, channel]

        # Ensure it's in 8-bit integer type
        gray_image = (channel_image * 255).astype(np.uint8)

        # Calculate GLCM
        glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=levels, symmetric=False, normed=True)

        for angle_idx in range(len(angles)):
            # Accumulate co-occurrence matrices
            co_occurrence_array[:, :, channel] += glcm[:, :, 0, angle_idx]

        # Normalize the accumulated co-occurrence matrix for each channel
        co_occurrence_array[:, :, channel] /= len(angles)

    return co_occurrence_array


def calculate_joint_entropy_red_green(img_arr):
    """
    Calculate the joint entropy of the red and green channels in the given image array.

    Parameters:
    - img_arr: 3D NumPy array representing the image. The first dimension is height,
               the second is width, and the third is channels (RGB).

    Returns:
    - Scalar value representing the joint entropy.
    """
    # Ensure the image values are integers.
    img_arr = np.array(img_arr, dtype=int)

    # Extract red and green channels from the image array
    red_channel, green_channel = img_arr[:, :, 0], img_arr[:, :, 1]

    # Calculate the 2D histogram
    joint_histogram, _, _ = np.histogram2d(red_channel.ravel(), green_channel.ravel(), bins=256,
                                           range=[[0, 256], [0, 256]])

    # Calculate joint probabilities
    joint_probabilities = joint_histogram / joint_histogram.sum()

    # Filter out zero probabilities for entropy calculation
    non_zero_probs = joint_probabilities[joint_probabilities > 0]

    # Calculate joint entropy
    joint_entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))

    return joint_entropy


def calculate_joint_RGB_entropy(rgb_image):
    """
    Calculate the joint entropy for the RGB channels in the given image array.

    Parameters:
    - rgb_image: 3D NumPy array representing the image. The first dimension is height,
                 the second is width, and the third is channels (RGB).

    Returns:
    - Scalar representing the joint entropy.
    """
    # Ensure the image values are integers.
    rgb_image = np.array(rgb_image, dtype=int)

    # Flatten and stack the color channels
    rgb_flatten = np.vstack([rgb_image[:, :, i].ravel() for i in range(3)]).T

    # Calculate the 3D histogram
    joint_histogram, _ = np.histogramdd(rgb_flatten, bins=256, range=[[0, 256], [0, 256], [0, 256]])

    # Calculate joint probabilities
    joint_probabilities = joint_histogram / joint_histogram.sum()

    # Filter out zero probabilities for entropy calculation
    non_zero_probs = joint_probabilities[joint_probabilities > 0]

    # Calculate joint entropy
    joint_entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))

    return joint_entropy


def calculate_texture_entropy(img_arr):
    """
    Calculate the histogram of Local Binary Pattern (LBP) values for the texture of a given image.

    Parameters:
    - img_arr: 2D or 3D NumPy array representing the grayscale or color image.

    Returns:
    - 1D NumPy array representing the histogram of LBP values.
    """
    # Convert the image to grayscale if it's a color image
    if img_arr.ndim == 3 and img_arr.shape[-1] == 3:
        gray_image = rgb2gray(img_arr)
    elif img_arr.ndim == 2 or (img_arr.ndim == 3 and img_arr.shape[-1] == 1):
        gray_image = img_arr.squeeze()
    else:
        raise ValueError("Input image should be either grayscale or RGB.")

    # Normalize the grayscale image if it isn't already
    if gray_image.max() > 1:
        gray_image /= 255.0

    # Apply Local Binary Pattern (LBP) to extract texture features
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(gray_image, n_points, radius, method='uniform')

    # Calculate histogram of LBP values
    n_bins = int(n_points * (n_points - 1) / 2) + 2
    hist, _ = np.histogram(lbp_image, bins=n_bins, range=(0, n_bins))

    # Normalize histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + np.finfo(float).eps)

    return hist


def calculate_texture_gabor_entropy(img_arr):
    """
        Calculate the histogram of Gabor-filtered values for the texture of a given image.

        Parameters:
        - img_arr: 2D or 3D NumPy array representing the grayscale or color image.

        Returns:
        - 1D NumPy array representing the histogram of Gabor-filtered values.
        """
    # Convert the image to grayscale if it's a color image
    if img_arr.ndim == 3 and img_arr.shape[-1] in [3, 4]:
        gray_image = 0.299 * img_arr[:, :, 0] + 0.587 * img_arr[:, :, 1] + 0.114 * img_arr[:, :, 2]
    elif img_arr.ndim == 2 or (img_arr.ndim == 3 and img_arr.shape[-1] == 1):
        gray_image = img_arr.squeeze()
    else:
        raise ValueError("Input image should be either grayscale or RGB.")

    # Define Gabor filter parameters
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

    # Calculate histogram
    hist, _ = np.histogram(gabor_response, bins=256, density=True)

    return hist


def adaptive_entropy_estimation(img_arr, num_segments=100):
    """
        Estimate the adaptive entropy of an image by segmenting it and averaging the entropies of the segments.

        Parameters:
        - img_arr: 2D or 3D NumPy array representing the image.
        - num_segments: Number of segments to divide the image into using the SLIC algorithm.

        Returns:
        - float: Adaptive entropy of the image.
        """
    # Convert the image to grayscale if it's a color image
    if img_arr.ndim == 3:
        gray_image = rgb2gray(img_arr)
    else:
        gray_image = img_arr

    # Segment the image using SLIC
    segments = slic(img_arr, n_segments=num_segments, compactness=10, sigma=1)

    # Initialize list to store segment entropies
    segment_entropies = []

    # Loop through each unique segment
    unique_segments = np.unique(segments)
    for segment_idx in unique_segments:
        segment_mask = (segments == segment_idx)
        segment_region = gray_image[segment_mask]

        # Calculate the entropy of each segment using the Shannon entropy formula
        hist, _ = np.histogram(segment_region, bins=256)
        prob_dist = hist / hist.sum()
        segment_entropy = entropy(prob_dist)

        segment_entropies.append(segment_entropy)

    # Calculate adaptive entropy
    adaptive_entropy = np.mean(segment_entropies)

    return adaptive_entropy


def laplace_ent(image):
    return laplacian(image)


def batch_entropy_calculation(methods, img_arr, ent_norm=None, color_weight=None, sortbymethod=None):
    """
    Computes entropy for a batch of images using multiple methods.

    Args:
        methods (list): List of entropy calculation methods as strings.
        img_arr (list): List of image arrays.
        ent_norm (dict, optional): Normalization dictionary to normalize the entropy based on a fixed image.
        color_weight: Optional parameter, this will be used to determine the RGB contribution in entropy calculation.
        sortbymethod (str): Optional parameter, this will be used to sort the result according to the method, if not given, it is not sorted
    Returns:
        list: A list of lists, where each inner list contains an image array as the first element and a dictionary of
              calculated entropies as the second element.
    """
    results = []

    for img in img_arr:
        ent_dict = {}

        for method in methods:
            ent_value = calc_ent(img, method, ent_norm, color_weight)

            if ent_value is not None:  # Checking in case calc_ent returns None for an unrecognized method
                ent_dict[method] = ent_value

        results.append([img, ent_dict])
    if sortbymethod is not None:
        return sort_by_entropy(results, sortbymethod)
    return results


def sort_by_entropy(batch_result, method):
    """
    Sorts the result of batch_entropy_calculation based on the entropy value calculated by a specific method.

    Args:
        batch_result (list): The list of lists returned by batch_entropy_calculation.
        method (str): The method whose entropy values should be used for sorting.

    Returns:
        list: A list of lists sorted based on the entropy value calculated by the specified method.
    """

    # Check if the method exists in the first dictionary to avoid sorting by a non-existing method
    if method not in batch_result[0][1]:
        print(f"Method {method} not found in the entropy dictionary.")
        return None

    # Sort based on the entropy value for the specified method
    sorted_result = sorted(batch_result, key=lambda x: x[1].get(method, 0))

    return sorted_result


def complexity(s_dict, method=None):
    c = 0
    if isinstance(method, dict):
        # calculate by weights with linear combination
        for met in s_dict:
            c += s_dict[met] * method[met]
    return c
