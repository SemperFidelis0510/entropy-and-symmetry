import numpy as np
import cv2
from scipy.stats import entropy


def calc_ent(img_arr, method):
    img_entropy = 0
    match method:
        case 'hist':
            bins_ = 256
            hists,bins = np.histogram(img_arr.ravel(), bins=bins_, range=(0, bins_))
            hists_done = hists / hists.sum()
            img_entropy = entropy(hists_done)
    return img_entropy


image_path = '/Users/tangjingqin/Desktop/entropy/entropy-and-symmetry/datasets/hilbertCurve2.png'
img_arr = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
print(calc_ent(img_arr, 'hist'))
