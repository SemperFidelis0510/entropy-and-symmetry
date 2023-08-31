import numpy as np
from source.Image import Image
import pywt
from skimage.color import rgb2gray
from scipy.ndimage import convolve
from skimage.feature import graycomatrix
from skimage.feature import local_binary_pattern
from scipy.signal import convolve2d
from skimage.segmentation import slic


class Processor:
    def __init__(self, processing_methods_with_params=None, level=0):
        self.processing_methods_with_params = processing_methods_with_params
        self.level = level
    def applyProcessing(self, image: Image):
        for method, params in self.processing_methods_with_params.items():
            if method == 'dft':
                image.processedData[method] = self.apply_dft(image)
            elif method == 'dwt':
                image.processedData[method] = self.apply_dwt(image.preprocessedData, **params)
            elif method == 'naive':
                image.processedData[method] = self.apply_naive(image)
            elif method == 'hist':
                image.processedData[method] = self.apply_histogram(image.preprocessedData)
            elif method == 'laplace':
                image.processedData[method] = self.apply_laplacian(image)
            elif method == 'joint_red_green':
                image.processedData[method] = self.apply_joint_red_green(image)
            elif method == 'joint_all':
                image.processedData[method] = self.apply_joint_RGB(image)
            elif method == 'lbp':
                image.processedData[method] = self.apply_texture(image)
            elif method == 'lbp_gabor':
                image.processedData[method] = self.apply_texture_gabor(image)
            elif method == 'adapt':
                image.processedData[method] = self.apply_adaptive_estimation(image, **params)
            elif method == 'RGBCM':
                image.processedData[method] = self.apply_CM_co_occurrence(image)

            else:
                raise ValueError(f"No entropy method matched for method '{method}'!!")

    def apply_naive(self, image):
        results = []
        for level in range(self.level+1):
            results.append(self.compute_naive(image, level))
        return results

    def compute_naive(self, image, level):
        partition_matrix = self.partition_image(image.preprocessedData, 2 ** level)
        return partition_matrix
    def apply_dft(self, image):
        results = []
        for level in range(self.level+1):
            results.append(self.compute_dft(image, level))
        return results
    def compute_dft(self, image, level):
        partition_matrix = self.partition_image(image.preprocessedData, 2**level)
        dft_matrix = []
        for partition_row in partition_matrix:
            dft_row = []
            for sub_image in partition_row:
                    #sub_image = partition_matrix[row][column]
                    result = np.empty_like(sub_image, dtype=np.float64)
                    for i in range(sub_image.shape[2]):
                        result[:, :, i] = np.abs(np.fft.fft2(sub_image[:, :, i]))
                    dft_row.append(result)
            dft_matrix.append(dft_row)
        return dft_matrix

    def apply_dwt(self, image, wavelet='db1', level=None):
        result = []
        if level != 'all':
            result.append(self.compute_dwt(image, wavelet=wavelet, level=level))
        else:
            result = self.compute_dwt(image, wavelet=wavelet, level=None)
        return result

    def compute_dwt(self, image, wavelet='db1', level=None):
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
            result = []
            for i in range(image.shape[2]):
                temp = pywt.wavedec2(image[:, :, i], wavelet=wavelet, level=level)
                result.append(temp)

            return result

        else:
            raise ValueError("Array must be 1D, 2D, or 3D")

    def apply_histogram(self, image):
        results = []
        for level in range(self.level+1):
            results.append(self.compute_histogram(image, level))
        return results
    def compute_histogram(self, image, level):
        partition_matrix = self.partition_image(image.preprocessedData, 2 ** level)
        hist_matrix = []
        for partition_row in partition_matrix:
            hist_row = []
            for sub_image in partition_row:
                sub_image = sub_image.astype(np.uint32)  # Convert to an integer type
                # Reduce color resolution by right-shifting
                reduced_img_arr = sub_image >> 4
                # Combine the reduced RGB values into a single integer
                flattened_img_arr = (reduced_img_arr[:, :, 0] << 12) + (reduced_img_arr[:, :, 1] << 6) + reduced_img_arr[:, :, 2]
                # Create the histogram with fewer bins
                bins_ = 64 ** 3
                hist, _ = np.histogram(flattened_img_arr, bins=bins_, range=(0, bins_ - 1))
                hist_row.append(hist)
            hist_matrix.append(hist_row)
        return hist_matrix

    def apply_laplacian(self, image):
        results = []
        for level in range(self.level + 1):
            results.append(self.compute_laplacian(image, level))
        return results
    def compute_laplacian(self, image, level):
        partition_matrix = self.partition_image(image.preprocessedData, 2 ** level)
        laplacian_matrix = []
        for partition_row in partition_matrix:
            laplacian_row = []
            for sub_image in partition_row:
                kernel = np.zeros((3, 3, 3))
                kernel[1, 1, 1] = 6
                kernel[1, 1, 0] = kernel[1, 1, 2] = kernel[1, 0, 1] = kernel[1, 2, 1] = kernel[0, 1, 1] = kernel[
                    2, 1, 1] = -1  # Apply the convolution with the kernel
                result = convolve(sub_image, kernel, mode='constant', cval=0.0)
                laplacian_row.append(result)
            laplacian_matrix.append(laplacian_row)
        return laplacian_matrix

    def apply_joint_red_green(self, image):
        results = []
        for level in range(self.level + 1):
            results.append(self.compute_joint_red_green(image, level))
        return results
    def compute_joint_red_green(self, image, level):
        partition_matrix = self.partition_image(image.preprocessedData, 2 ** level)
        rg_matrix = []
        for partition_row in partition_matrix:
            rg_row = []
            for sub_image in partition_row:
                # Extract red and green channels from the image array
                red_channel, green_channel = sub_image[:, :, 0], sub_image[:, :, 1]

                # Calculate the 2D histogram
                joint_histogram, _, _ = np.histogram2d(red_channel.ravel(), green_channel.ravel(), bins=256)

                # Calculate joint probabilities
                joint_probabilities = joint_histogram / joint_histogram.sum()
                rg_row.append(joint_probabilities)
            rg_matrix.append(rg_row)
        return rg_matrix

    def apply_joint_RGB(self, image):
        results = []
        for level in range(self.level+1):
            results.append(self.compute_joint_RGB(image, level))
        return results
    def compute_joint_RGB(self, image, level):
        partition_matrix = self.partition_image(image.preprocessedData, 2**level)
        processed_matrix = []
        for partition_row in partition_matrix:
            processed_row = []
            for sub_image in partition_row:
                # Flatten and stack the color channels
                rgb_flatten = np.vstack([sub_image[:, :, i].ravel() for i in range(3)]).T
                # Calculate the 3D histogram
                joint_histogram, _ = np.histogramdd(rgb_flatten, bins=256, range=[[0, 256], [0, 256], [0, 256]])
                # Calculate joint probabilities
                joint_probabilities = joint_histogram / joint_histogram.sum()
                processed_row.append(joint_probabilities)
            processed_matrix.append(processed_row)
        return processed_matrix
    def apply_texture(self, image):
        results = []
        for level in range(self.level+1):
            results.append(self.compute_texture(image, level))
        return results
    def compute_texture(self, image, level):
        partition_matrix = self.partition_image(image.preprocessedData, 2 ** level)
        texture_matrix = []
        for partition_row in partition_matrix:
            texture_row = []
            for sub_image in partition_row:
                # Convert the image to grayscale if it's a color image
                if sub_image.ndim == 3 and sub_image.shape[-1] == 3:
                    gray_image = rgb2gray(sub_image)
                elif sub_image.ndim == 2 or (sub_image.ndim == 3 and sub_image.shape[-1] == 1):
                    gray_image = sub_image.squeeze()
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
                texture_row.append(hist)
            texture_matrix.append(texture_row)
        return texture_matrix



    def apply_texture_gabor(self, image):
        results = []
        for level in range(self.level + 1):
            results.append(self.compute_texture_gabor(image, level))
        return results

    def compute_texture_gabor(self, image, level):
        partition_matrix = self.partition_image(image.preprocessedData, 2 ** level)
        tg_matrix = []
        for partition_row in partition_matrix:
            tg_row = []
            for sub_image in partition_row:
                # Convert the image to grayscale if it's a color image
                if sub_image.ndim == 3 and sub_image.shape[-1] in [3, 4]:
                    gray_image = 0.299 * sub_image[:, :, 0] + 0.587 * sub_image[:, :, 1] + 0.114 * sub_image[:, :, 2]
                elif sub_image.ndim == 2 or (sub_image.ndim == 3 and sub_image.shape[-1] == 1):
                    gray_image = sub_image.squeeze()
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
                tg_row.append(hist)
            tg_matrix.append(tg_row)
        return tg_matrix



    def apply_adaptive_estimation(self, image, num_segments=100):
        results = []
        for level in range(self.level+1):
            results.append(self.compute_adaptive_estimation(image, level, num_segments))
        return results
    def compute_adaptive_estimation(self, image, level, num_segments):
        partition_matrix = self.partition_image(image.preprocessedData, 2**level)
        processed_matrix = []
        for partition_row in partition_matrix:
            processed_row = []
            for sub_image in partition_row:
                # Convert the image to grayscale if it's a color image
                if sub_image.ndim == 3:
                    gray_image = rgb2gray(sub_image)
                else:
                    gray_image = sub_image

                # Segment the image using SLIC
                segments = slic(sub_image, n_segments=num_segments, compactness=10, sigma=1)

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
                processed_row.append(segment)
            processed_matrix.append(processed_row)
        return processed_matrix



    def apply_CM_co_occurrence(self, image):
        results = []
        for level in range(self.level+1):
            results.append(self.compute_CM_co_occurrence(image, level))
        return results
    def compute_CM_co_occurrence(self, image, level):
        partition_matrix = self.partition_image(image.preprocessedData, 2 ** level)
        CM_matrix = []
        for partition_row in partition_matrix:
            CM_row = []
            for sub_image in partition_row:
                distances = [1]  # Distance between pixels for co-occurrence
                angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]  # Angles for co-occurrence (in radians)
                levels = 256  # Number of intensity levels in the image

                # Initialize array for co-occurrence matrices
                co_occurrence_array = np.zeros((levels, levels, 3))

                for channel in range(3):  # Iterate over RGB channels
                    channel_image = sub_image[:, :, channel]

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
                CM_row.append(co_occurrence_array)
            CM_matrix.append(CM_row)
        return CM_matrix


    def partition_image(self, image, partition):
        """
        Returns:
            list: A 2D list (matrix) where each element is a sub-image.
        """
        # Get the shape of the image
        height, width, _ = image.shape
        # Calculate the size of each partition
        partition_height = height // partition
        partition_width = width // partition

        # Initialize the result matrix
        result = []

        for i in range(0, height, partition_height):
            row = []
            for j in range(0, width, partition_width):
                # Extract the sub-image
                sub_image = image[i:i + partition_height, j:j + partition_width]
                row.append(sub_image)
            result.append(row)

        return result
