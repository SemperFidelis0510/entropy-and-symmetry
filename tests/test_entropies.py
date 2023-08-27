import sys
import numpy as np
from skimage.color import rgb2gray
from scripts.entropy import (
    entropy,
    histogram,
    calculate_CM_co_occurrence,
    calculate_joint_entropy_red_green,
    calculate_joint_RGB_entropy,
    calculate_texture_entropy,
    calculate_texture_gabor_entropy,
    adaptive_entropy_estimation,
    laplace_ent,
)

# Generate a random image for testing
test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
test_gray_image = rgb2gray(test_image)

# Explicitly cast dtype to int
test_image_int = test_image.astype(int)


def run_tests():
    assert isinstance(entropy(test_gray_image), float), "Error in test_entropy"
    assert isinstance(histogram(test_image), np.ndarray), "Error in test_histogram"
    assert isinstance(calculate_CM_co_occurrence(test_image), np.ndarray), "Error in test_calculate_CM_co_occurrence"
    assert isinstance(calculate_joint_entropy_red_green(test_image_int),
                      float), "Error in test_calculate_joint_entropy_red_green"
    assert isinstance(calculate_joint_RGB_entropy(test_image), float), "Error in test_calculate_joint_RGB_entropy"
    assert isinstance(calculate_texture_entropy(test_gray_image), np.ndarray), "Error in test_calculate_texture_entropy"
    assert isinstance(calculate_texture_gabor_entropy(test_gray_image),
                      np.ndarray), "Error in test_calculate_texture_gabor_entropy"
    assert isinstance(adaptive_entropy_estimation(test_image), float), "Error in test_adaptive_entropy_estimation"
    assert isinstance(laplace_ent(test_gray_image), np.ndarray), "Error in test_laplace_ent"
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
