from functions import *
import os
from datetime import datetime


def transform(img):
    return img


def main():
    folder_path = ''
    method = 'hist'
    dst_folder = f'processed/{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    list_of_paths = load_images(folder_path)
    img_list = []

    for img_path in list_of_paths:
        img = preprocess(img_path)
        img = transform(img)
        s = calc_ent(img, method)
        img_list.append([img, s])

    sorted_list = sorted(img_list, key=lambda x: x[1])

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    i = 0
    for obj in sorted_list:
        save_img(os.path.join(dst_folder, f'{i}.bmp'), obj[0])
        i += 1




if __name__ == '__main__':
    main()
