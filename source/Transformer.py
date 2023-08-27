import numpy as np
from Image import Image
import pywt
from skimage.color import rgb2gray
from scipy.ndimage import convolve
from skimage.feature import graycomatrix
from skimage.feature import local_binary_pattern
from scipy.signal import convolve2d
from skimage.segmentation import slic

class Transformer:
    def __init__(self, transformation_methods_with_params = None):
        self.transformation_methods_with_params = transformation_methods_with_params

    def applyTransform(self, image: Image):
        for method, params in self.transformation_methods_with_params.items():
            if method == 'dft':
                image.transformedData[method] = self.apply_dft(image.preprocessedData)
            elif method == 'dwt':
                image.transformedData[method] = self.apply_dwt(image.preprocessedData, **params)
            elif method == 'naive':
                image.transformedData[method] = image.preprocessedData
            elif method == 'hist':
                image.transformedData[method] = self.apply_histogram(image.preprocessedData)
            elif method == 'laplace':
                image.transformedData[method] = self.apply_laplacian(image.preprocessedData)
            elif method == 'joint_red_green':
                image.transformedData[method] = self.apply_joint_red_green(image.preprocessedData)
            elif method == 'joint_all':
                image.transformedData[method] = self.apply_joint_RGB(image.preprocessedData)
            elif method == 'lbp':
                image.transformedData[method] = self.apply_texture(image.preprocessedData)

            elif method == 'lbp_gabor':
                image.transformedData[method] = self.apply_texture_gabor(image.preprocessedData)

            elif method == 'adapt':
                image.transformedData[method] = self.apply_adaptive_estimation(image.preprocessedData, **params)
            elif method == 'RGBCM':
                image.transformedData[method] = self.apply_CM_co_occurrence(image.preprocessedData)

            else:
                raise ValueError(f"No entropy method matched for method '{method}'!!")

    def apply_dft(self, image):
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

    def apply_dwt(self, image, wavelet='db1', level=None):
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

    def apply_histogram(self, img_arr):
        if img_arr.ndim == 3:  # Check if the tensor is RGB (rank 3)
            img_arr = img_arr.astype(np.uint32)  # Convert to an integer type
            # Reduce color resolution by right-shifting
            reduced_img_arr = img_arr >> 4
            # Combine the reduced RGB values into a single integer
            flattened_img_arr = (reduced_img_arr[:, :, 0] << 12) + (reduced_img_arr[:, :, 1] << 6) + reduced_img_arr[:,
                                                                                                     :,
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

    def apply_laplacian(self, arr):
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
            kernel[1, 1, 0] = kernel[1, 1, 2] = kernel[1, 0, 1] = kernel[1, 2, 1] = kernel[0, 1, 1] = kernel[
                2, 1, 1] = -1
        else:
            raise ValueError("Array must be 1D, 2D, or 3D")

        # Apply the convolution with the kernel
        result = convolve(arr, kernel, mode='constant', cval=0.0)

        return result

    def apply_joint_red_green(self, img_arr):
        # Extract red and green channels from the image array
        red_channel, green_channel = img_arr[:, :, 0], img_arr[:, :, 1]

        # Calculate the 2D histogram
        joint_histogram, _, _ = np.histogram2d(red_channel.ravel(), green_channel.ravel(), bins=256)

        # Calculate joint probabilities
        joint_probabilities = joint_histogram / joint_histogram.sum()

        return joint_probabilities

    def apply_joint_RGB(self, rgb_image):
        # Flatten and stack the color channels
        rgb_flatten = np.vstack([rgb_image[:, :, i].ravel() for i in range(3)]).T

        # Calculate the 3D histogram
        joint_histogram, _ = np.histogramdd(rgb_flatten, bins=256, range=[[0, 256], [0, 256], [0, 256]])

        # Calculate joint probabilities
        joint_probabilities = joint_histogram / joint_histogram.sum()

        return joint_probabilities

    def apply_texture(self, img_arr):
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

    def apply_texture_gabor(self, img_arr):
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

    def apply_adaptive_estimation(self, img_arr, num_segments=100):
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
        segment = []

        # Loop through each unique segment
        unique_segments = np.unique(segments)
        for segment_idx in unique_segments:
            segment_mask = (segments == segment_idx)
            segment_region = gray_image[segment_mask]

            # Calculate the entropy of each segment using the Shannon entropy formula
            hist, _ = np.histogram(segment_region, bins=256)
            prob_dist = hist / hist.sum()
            segment.append(prob_dist)

        return segment

    def apply_CM_co_occurrence(self, image):
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
            glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=levels, symmetric=False,
                                normed=True)

            for angle_idx in range(len(angles)):
                # Accumulate co-occurrence matrices
                co_occurrence_array[:, :, channel] += glcm[:, :, 0, angle_idx]

            # Normalize the accumulated co-occurrence matrix for each channel
            co_occurrence_array[:, :, channel] /= len(angles)

        return co_occurrence_array