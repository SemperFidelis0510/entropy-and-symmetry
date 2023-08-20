import json
from datetime import datetime

from entropy import *
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


def norm_ent():
    ENTROPY_METHODS = [
        'hist',
        'hist_greyscale',
        'naive',
        'dft',
        'dwt',
        'laplace',
        'joint_red_green',
        'joint_all',
        'lbp',
        'lbp_gabor',
        'adapt',
        'GLCM',
        'RGBCM_each_channel',
        'RGBCM_to_gray'
    ]
    ENTROPY_METHODS = [
        'lbp'
    ]

    path = '../datasets/fixed_noise.bmp'
    img_arr = np.array(Image.open(path))
    N = {}
    for ent in ENTROPY_METHODS:
        N[ent] = calc_ent(img_arr, ent)

    with open('ent_norm', 'w') as file:
        json.dump(N, file)


def main():
    # folder_path = r'../datasets/pattern_images'
    folder_path = r'../datasets/web_symmetries/rotation'
    # folder_path = r'../datasets/Fractals with controlled parameter/Line-Replacement Fractals/snowflake'
    folder_path = '../datasets/noising'

    folder_path = normalize_path(folder_path)
    # print(f'Dataset path: {os.path.abspath(folder_path)}')

    # sort_folder(folder_path)
    # sort_by_noise(folder_path)

    norm_ent()


if __name__ == '__main__':
    main()
