import numpy as np
import cv2
from scipy.stats import entropy


def calc_ent(img_arr, method):
    img_entropy = 0
    match method:
        case 'hist':
            bins_ = 256
            hists, bins = np.histogram(img_arr.ravel(), bins=bins_, range=(0, bins_))
            hists_done = hists / hists.sum()
            img_entropy = entropy(hists_done)
    return img_entropy


def load_images(path):
    pass


def preprocess(path):
    pass
