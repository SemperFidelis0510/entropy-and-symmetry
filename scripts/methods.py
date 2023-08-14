import numpy as np
from skimage.feature import local_binary_pattern
from skimage import color
from skimage.filters import gabor
from skimage.color import rgb2gray

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
    if len(image.shape) == 3:
        gray_image = rgb2gray(image)
    else:
        gray_image = image
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
    # Convert the image to grayscale if it's not already
    if len(img_arr.shape) == 3:
        gray_img_arr = rgb2gray(img_arr)
    else:
        gray_img_arr = img_arr

    # Apply Gabor filters to extract texture features
    frequency = 0.6
    theta = np.arange(0, np.pi, np.pi / 4)  # Orientations: 0, 45, 90, 135 degrees
    gabor_responses = []
    for angle in theta:
        kernel = np.real(gabor(gray_img_arr, frequency, theta=angle)[0])  # Take the real part of the Gabor kernel
        response = np.abs(np.convolve(gray_img_arr.ravel(), kernel.ravel(), mode='same').reshape(gray_img_arr.shape))
        gabor_responses.append(response)
    gabor_responses = np.array(gabor_responses)

    # Calculate LBP of the Gabor filter responses
    lbp_responses = []
    for response in gabor_responses:
        lbp_image = local_binary_pattern(response, 8, 1, method='uniform')
        lbp_responses.append(lbp_image)
    lbp_responses = np.array(lbp_responses)

    # Calculate histogram of LBP values
    hist, _ = np.histogram(lbp_responses.ravel(), bins=np.arange(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + np.finfo(float).eps)

    # Calculate entropy of LBP histogram
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))

    return entropy

def calculate_rgb_color_cube_entropy(rgb_image, num_bins=8):
    # Ensure the number of bins is a power of 2 for simplicity
    if not (num_bins > 0 and (num_bins & (num_bins - 1)) == 0):
        raise ValueError("Number of bins must be a power of 2")

    # Normalize the RGB values to the range [0, num_bins)
    normalized_rgb = (rgb_image * num_bins / 256).astype(int)

    # Calculate the histogram of normalized RGB values
    hist = np.histogramdd(normalized_rgb.reshape(-1, 3), bins=(num_bins, num_bins, num_bins), range=((0, num_bins), (0, num_bins), (0, num_bins)))[0]

    # Normalize the histogram to obtain a probability distribution
    p = hist / np.sum(hist)

    # Calculate entropy using the definition of entropy
    entropy = -np.sum(p * np.log2(p + np.finfo(float).eps))

    return entropy

