from functions import *
import os
from datetime import datetime


def main():
    folder_path = r'C:\scripts\entropy-and-symmetry\datasets\pattern_images'
    method = 'hist'
    dst_folder = f'../processed/{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    list_of_paths = load_images(folder_path)
    img_list = []

    for img_path in list_of_paths:
        img_list.append(preprocess(img_path))

    sorted_list = sorted(img_list, key=lambda x: calc_ent(x, method))
    # sorted_list = img_list

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    i = 0
    for obj in sorted_list:
        save_img(os.path.join(dst_folder, f'{i}.bmp'), obj)
        i += 1


if __name__ == '__main__':
    main()
