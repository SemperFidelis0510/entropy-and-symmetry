import numpy as np
from Image import Image
import pywt
class Transformer:
    def __init__(self, transformation_methods_with_params = None):
        self.transformation_methods_with_params = transformation_methods_with_params

    def applyTransform(self, image: Image):
        for method, params in self.transformation_methods_with_params.items():
            if method == 'dft':
                image.transformedData.append(self.apply_dft(image.preprocessedData))
            elif method == 'dwt':
                image.transformedData.append(self.apply_dwt(image.preprocessedData, **params))
            elif method == 'naive':
                image.transformedData.append(image.preprocessedData)
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