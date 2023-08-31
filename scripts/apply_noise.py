import random
from datetime import datetime

from functions import *


def apply_noise(img_arr):
    new_imgs = []
    dst_folder = f'../processed/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    num_of_images = random.randint(1, 100)
    for i in range(1, num_of_images + 1):
        level = random.random()
        new_img_arr = uniform_noise(img_arr, level)
        new_imgs.append(new_img_arr)
    save_img(dst_folder, new_imgs)
    return
