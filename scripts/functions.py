import numpy as np
import cv2
from scipy.stats import entropy
from PIL import Image
import os

def calc_ent(img_arr, method):
    img_entropy = 0
    match method:
        case 'hist':
            bins_ = 256
            hists, bins = np.histogram(img_arr.ravel(), bins=bins_, range=(0, bins_))
            hists_done = hists / hists.sum()
            img_entropy = entropy(hists_done)
    return img_entropy


def save_img(path, arr):
    img = Image.fromarray(arr)
    img.save(path)


def load_images(path):
    imgs_path = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        imgs_path.append(img_path)
    return imgs_path


def preprocess(path, crop_size=None):
    img = Image.open(path)
    if crop_size == None:
        crop_size = min(img.size)
    cropped = img.crop((0,0,crop_size,crop_size))
    img_arr = np.array(cropped)
    return img_arr
