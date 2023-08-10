import numpy as np
from scipy.stats import entropy


def calc_ent(img_arr, method):
    img_entropy = 0
    match method:
        case 'hist_grey':
            bins_ = 256
            hist, bins = np.histogram(img_arr.ravel(), bins=bins_, range=(0, bins_ - 1))
            hist_done = hist / hist.sum()
            img_entropy = entropy(hist_done)
        case 'hist_RGB':
            bins_ = 256 ** 3
            flattened_img_arr = (img_arr[:, :, 0] << 16) + (img_arr[:, :, 1] << 8) + img_arr[:, :, 2]
            hist, _ = np.hisogram(flattened_img_arr, bins=bins_, range=(0, bins_ - 1))
            hist_done = hist / np.sum(hist)
            img_entropy = -np.sum(hist_done * np.log(hist_done + np.finfo(float).eps))

    return img_entropy


