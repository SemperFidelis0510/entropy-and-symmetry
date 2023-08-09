import numpy as np
from scipy.stats import entropy


def calc_ent(img_arr, method):
    img_entropy = 0
    match method:
        case 'hist_grey':
            bins_ = 256
            hists, bins = np.histogram(img_arr.ravel(), bins=bins_, range=(0, bins_))
            hists_done = hists / hists.sum()
            img_entropy = entropy(hists_done)
        case 'hist_RGB':
            bins_ = 256
            r = img_arr[:, :, 0]
            g = img_arr[:, :, 1]
            b = img_arr[:, :, 2]
            total = np.concatenate((r, g, b))
            hists_RGB, bins = np.histogram(total, bins=bins_, range=(0, bins_))
            hists_RGB_done = hists_RGB / hists_RGB.sum()
            img_entropy = entropy(hists_RGB_done)
    return img_entropy


