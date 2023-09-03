import numpy as np
from source.Image import Image
import pywt
from skimage.color import rgb2gray
from skimage.feature import graycomatrix
from skimage.feature import local_binary_pattern
from pytorch_wavelets import DWTForward
from skimage.segmentation import slic
import torch
import torch.nn.functional as F
from PIL import Image as PILImage
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
                image.processedData[method] = self.apply_histogram(image)
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
        processed_matrix = []
        for partition_row in partition_matrix:
            processed_row = []
            for sub_image in partition_row:
                img = self.apply_ycrcb(sub_image)
                processed_row.append(img)
            processed_matrix.append(processed_row)
        return processed_matrix
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
                sub_image = self.apply_ycrcb(sub_image)
                result = torch.empty_like(sub_image, dtype=torch.float64)
                # Ensure the tensor is on the same device (GPU) as img
                result = result.to(sub_image.device)
                for i in range(sub_image.shape[0]):
                    channel_data = sub_image[i,:,:]
                    fft_result = torch.fft.fft2(channel_data)
                    result[i,:,:] = torch.abs(fft_result)
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
        image = self.apply_ycrcb(image)
        # Calculate the maximum number of decomposition levels
        if level is None:
            J_level = 9 #int(torch.floor(torch.log2(torch.min(torch.tensor([image.shape[1], image.shape[2]], dtype=torch.float32)))).item())
        else:
            J_level = level
        dwt = DWTForward(wave=wavelet, J=J_level).cuda()

        result = []
        for i in range(image.shape[0]):  # Iterate over Y,Cb,Cr channels
            channel_image = image[i, :, :].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

            # Calculate DWT using pytorch-wavelets
            Yl, Yh = dwt(channel_image)
            result.append((Yl, Yh))

        return result

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
                # Assuming sub_image is a PyTorch tensor on GPU with shape (C, H, W)

                # Scale the values to 0-15 range for right-shifting
                scaled_sub_image = (sub_image * 255).to(torch.uint8)

                # Reduce color resolution by right-shifting (in-place operation)
                reduced_img_tensor = scaled_sub_image >> 4

                # Combine the reduced RGB values into a single integer
                # Using bitwise operations directly on slices of the original tensor
                flattened_img_tensor = (reduced_img_tensor[0] << 8) | (reduced_img_tensor[1] << 4) | reduced_img_tensor[
                    2]

                # Create the histogram with fewer bins
                bins_per_channel = 64  # Number of bins per color channel
                bins_total = bins_per_channel ** 3

                # Flatten the tensor and calculate histogram
                flattened_img_tensor = flattened_img_tensor.view(-1)
                hist = torch.histc(flattened_img_tensor.float(), bins=bins_total, min=0, max=bins_total - 1)
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
                # Create the kernel as a PyTorch tensor
                kernel = torch.zeros(3, 3, 3)
                kernel[1, 1, 1] = 6
                kernel[1, 1, 0] = kernel[1, 1, 2] = kernel[1, 0, 1] = kernel[1, 2, 1] = kernel[0, 1, 1] = kernel[
                    2, 1, 1] = -1

                # Move the kernel to the same device as sub_image
                kernel = kernel.to(sub_image.device)

                # Add an extra batch dimension and an extra channel dimension to both the image and the kernel
                sub_image = sub_image.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, C, H, W]
                kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape becomes [1, 1, 3, 3, 3]

                # Apply the 3D convolution
                result = F.conv3d(sub_image, kernel, padding=1)

                # Remove the extra dimensions
                result = result.squeeze(0).squeeze(0)
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

                red_channel, green_channel = sub_image[0, :, :], sub_image[1, :, :]

                # Flatten the channels
                red_channel_flat = red_channel.reshape(-1)
                green_channel_flat = green_channel.reshape(-1)

                # Calculate the 2D histogram
                joint_histogram = torch.histc(torch.stack((red_channel_flat, green_channel_flat), dim=1), bins=256,
                                              min=0, max=1)

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
        partition_matrix = self.partition_image(torch.tensor(image.preprocessedData), 2**level)
        processed_matrix = []
        for partition_row in partition_matrix:
            processed_row = []
            for sub_image in partition_row:
                # Stack all three RGB channels
                rgb_channels = sub_image[0:3, :, :]


                # Flatten the normalized channels
                rgb_channels_flat = rgb_channels.reshape(3, -1)

                # Calculate the 3D histogram
                hist_3d = torch.histc(rgb_channels_flat, bins=256, min=0, max=1)

                # Calculate joint probabilities
                joint_probabilities = hist_3d / hist_3d.sum()
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
                if sub_image.ndim == 3 and sub_image.shape[0] == 3:
                    gray_image = torch.einsum('kij,k->ij', sub_image, torch.tensor([0.2989, 0.5870, 0.1140]).to(sub_image.device))
                elif sub_image.ndim == 2 or (sub_image.ndim == 3 and sub_image.shape[0] == 1):
                    gray_image = sub_image.squeeze()
                else:
                    raise ValueError("Input image should be either grayscale or RGB.")

                # Normalize the grayscale image if it isn't already
                if gray_image.max() > 1:
                    gray_image /= 255.0

                # Apply Local Binary Pattern (LBP) to extract texture features
                radius = 1
                n_points = 8 * radius
                lbp_image = local_binary_pattern(gray_image.cpu().numpy(), n_points, radius, method='uniform')

                # Convert to PyTorch tensor and move to GPU
                lbp_image = torch.tensor(lbp_image, dtype=torch.float32).to(sub_image.device)

                # Calculate histogram of LBP values
                n_bins = int(n_points * (n_points - 1) / 2) + 2
                hist = torch.histc(lbp_image, bins=n_bins, min=0, max=n_bins)

                # Normalize histogram
                hist = hist.float()
                hist /= (hist.sum() + torch.finfo(torch.float32).eps)
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
                if sub_image.shape[0] in [3, 4]:
                    gray_image = 0.299 * sub_image[0, :, :] + 0.587 * sub_image[1, :, :] + 0.114 * sub_image[2, :, :]
                elif sub_image.shape[0] == 1:
                    gray_image = sub_image.squeeze(0)
                else:
                    raise ValueError("Input image should be either grayscale or RGB.")

                    # Define Gabor filter parameters
                wavelength = 5.0
                orientation = np.pi / 4
                frequency = 1 / wavelength
                sigma = 1.0

                # Create Gabor filter using PyTorch
                x, y = torch.meshgrid(torch.arange(-15, 16).float().to(sub_image.device),
                                      torch.arange(-15, 16).float().to(sub_image.device))
                gabor_real = torch.exp(-0.5 * (x ** 2 + y ** 2) / (sigma ** 2)) * torch.cos(
                    2 * np.pi * frequency * (
                                x * torch.cos(torch.tensor(orientation, device=sub_image.device)) + y * torch.sin(
                            torch.tensor(orientation, device=sub_image.device))))

                # Add extra dimensions to the Gabor filter and the image for batch and channel
                gabor_real = gabor_real.unsqueeze(0).unsqueeze(0)
                gray_image = gray_image.unsqueeze(0).unsqueeze(0)

                # Apply Gabor filter using PyTorch's conv2d
                gabor_response = F.conv2d(gray_image, gabor_real, padding=15)

                # Remove extra dimensions
                gabor_response = gabor_response.squeeze(0).squeeze(0)

                # Calculate histogram using PyTorch
                hist = torch.histc(gabor_response, bins=256, min=gabor_response.min(), max=gabor_response.max())
                tg_row.append(hist)
            tg_matrix.append(tg_row)
        return tg_matrix



    def apply_adaptive_estimation(self, image, num_segments=100):
        results = []
        for level in range(self.level+1):
            results.append(self.compute_adaptive_estimation(image, level, num_segments))
        return results

    def rgb_to_gray(self, rgb_image):
        r, g, b = rgb_image[0], rgb_image[1], rgb_image[2]
        gray_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray_image
    def compute_adaptive_estimation(self, image, level, num_segments):
        partition_matrix = self.partition_image(image.preprocessedData, 2**level)
        processed_matrix = []
        for partition_row in partition_matrix:
            processed_row = []
            for sub_image in partition_row:
                # Convert the image to grayscale if it's a color image
                if sub_image.dim() == 3:
                    gray_image = self.rgb_to_gray(sub_image)
                else:
                    gray_image = sub_image

                # Convert gray_image tensor to numpy array on CPU for SLIC
                gray_image_cpu = gray_image.cpu().numpy()

                # Segment the image using SLIC
                segments = slic(gray_image_cpu, n_segments=num_segments, compactness=10, sigma=1)

                # Initialize list to store segment entropies
                segment = []

                # Loop through each unique segment
                unique_segments = np.unique(segments)
                for segment_idx in unique_segments:
                    segment_mask = (segments == segment_idx)
                    segment_region = gray_image[segment_mask]
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
                # Assuming sub_image is a PyTorch tensor with shape [C, H, W] and is already on the GPU

                # Initialize parameters
                distances = [1]
                angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
                levels = 256

                # Initialize array for co-occurrence matrices
                co_occurrence_array = torch.zeros((levels, levels, 3), device=sub_image.device)

                # Move tensor to CPU for GLCM calculation
                sub_image_cpu = sub_image.cpu().numpy()

                for channel in range(3):  # Iterate over RGB channels
                    channel_image = sub_image_cpu[channel, :, :]

                    # Ensure it's in 8-bit integer type
                    gray_image = (channel_image * 255).astype(np.uint8)

                    # Calculate GLCM using scikit-image
                    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=levels, symmetric=False,
                                        normed=True)

                    for angle_idx in range(len(angles)):
                        # Accumulate co-occurrence matrices
                        co_occurrence_array[:, :, channel] += torch.tensor(glcm[:, :, 0, angle_idx],
                                                                           device=sub_image.device)

                    # Normalize the accumulated co-occurrence matrix for each channel
                    co_occurrence_array[:, :, channel] /= len(angles)
                CM_row.append(co_occurrence_array)
            CM_matrix.append(CM_row)
        return CM_matrix

    def partition_image(self, image, partition):
        # Get the shape of the image
        height = image.shape[1]
        width = image.shape[2]
        # Calculate the size of each partition
        partition_height = height // partition
        partition_width = width // partition

        # Initialize the result matrix
        result = []

        height_left = height % partition
        width_left = width % partition
        for i in range(0, height-height_left, partition_height):
            row = []
            for j in range(0, width-width_left, partition_width):
                if j//partition_width==partition - 1:
                    sub_image = image[:, i:i + partition_height, j:]
                elif i//partition_height==partition-1:
                    sub_image = image[:, i:, j:j + partition_width]
                else:
                    # Extract the sub-image
                    sub_image = image[:, i:i + partition_height, j:j + partition_width]
                row.append(sub_image)
            result.append(row)

        return result

    def apply_ycrcb(self, img):
        kR = 0.299
        kG = 0.587
        kB = 0.114
        R = img[0, :, :]
        G = img[1, :, :]
        B = img[2, :, :]
        y = kR * R + kG * G + kB * B
        cb = (-kR / (2 * (1 - kB))) * R + (-kG / (2 * (1 - kB))) * G + 1 / 2 * B
        cr = 1 / 2 * R + (-kG / (2 * (1 - kR))) * G + (-kB / (2 * (1 - kR))) * B
        # Scale and shift to 8-bit integer values
        y = torch.clamp((219 * y + 16), 16, 235)
        cb = torch.clamp((224 * cb + 128), 16, 240)
        cr = torch.clamp((224 * cr + 128), 16, 240)
        return torch.stack([y, cb, cr]).to(img.device)