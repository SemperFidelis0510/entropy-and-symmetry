import numpy as np
import pywt


def dft(image):
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
    height, width = im_arr.shape[:2]
    noise_arr = np.zeros((height, width), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            noise_arr[i][j] = np.random.uniform(-noise_level, noise_level)
    noise_arr = (im_arr + noise_arr) % 255
    return noise_arr


def noise_by_increment(im_arr, num_images):
    """
    This function generates a list of arrays of noised-up pictures with rising values of noise.
    Args:
        im_arr: image represented by an array
        num_images: Number of images to generate in the list
    Returns:
        A list of noised images represented by arrays
    """
    noised_images = []
    for i in range(num_images):
        noise_level = i / (num_images - 1) if num_images > 1 else 0
        noised_image = uniform_noise(im_arr, noise_level)
        noised_images.append(noised_image)
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
    """
    Compute wavelet transform of an image and save the approximation.

    Parameters:
        - image: 2D numpy array representing the image.
        - wavelet: Type of wavelet to be used. Default is 'db1' (Daubechies wavelet).
        - level: Level of decomposition. If None, max possible level is used.

    Returns:
        - coeffs: Wavelet coefficients.
    """

    # Decompose the image using discrete wavelet transform
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)

    return coeffs
