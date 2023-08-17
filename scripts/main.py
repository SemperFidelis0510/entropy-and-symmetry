from datetime import datetime

from functions import *


def sort_folder(path):
    # colors = 'greyscale'
    colors = 'rgb'
    method = 'dft'
    dst_folder = f'../processed/m={method}_t={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    img_arrays, _ = preprocess(path, colors=colors)

    sorted_list = label_ent(img_arrays, method, sort=True)

    print('\nThe images are sorted by entropy.')

    save_img(dst_folder, sorted_list)


def sort_by_noise(path):
    method = 'naive'
    dst_folder = f'../processed/noise_m={method}_t={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    n = 10

    img_arrays, _ = preprocess(path)
    img_arrays = noise_by_increment(img_arrays[0], n)

    sorted_list = label_ent(img_arrays, method, sort=True)

    print('\nThe images are sorted by entropy.')

    save_img(dst_folder, sorted_list)


def main():
    # folder_path = r'../datasets/pattern_images'
    folder_path = r'../datasets/web_symmetries/rotation'
    # folder_path = r'../datasets/Fractals with controlled parameter/Line-Replacement Fractals/snowflake'
    folder_path = '../datasets/noising'

    folder_path = normalize_path(folder_path)
    print(f'Dataset path: {os.path.abspath(folder_path)}')

    # sort_folder(folder_path)
    sort_by_noise(folder_path)


if __name__ == '__main__':
    main()
