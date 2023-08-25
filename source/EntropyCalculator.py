import numpy as np
from Image import Image
class EntropyCalculator:
    def __init__(self, color_weight = None):
        self.color_weight = color_weight or (0.2989, 0.5870, 0.1140)

    def calculateEntropy(self, image: Image):
        for transformedData in image.transformedData:
            arr = np.abs(transformedData)
            ent = 0
            if arr.ndim == 3:
                if arr.shape[-1] != 3:  # Check if the last dimension has 3 channels (RGB)
                    raise ValueError("entropy function: 3D array must represent an RGB image with three channels")
                arr = np.dot(arr, self.color_weight)
            total_sum = np.sum(arr)
            if total_sum == 0:
                return 0
            normalize_arr = arr / total_sum
            ent = -np.sum(normalize_arr * np.log2(normalize_arr + np.finfo(float).eps))
            image.entropyResults.append(ent)