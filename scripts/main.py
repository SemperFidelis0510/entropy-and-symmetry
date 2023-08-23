import json
from datetime import datetime

from functions import *

ent_methods = [
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
    'RGBCM_to_gray',
    'naive_hsb'
]


def sort_folder(path, method, colors='rgb', ent_norm=None, color_weight=None):
    # colors = 'greyscale'
    dst_folder = f'../processed/m={method}_t={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    img_arrays, _ = preprocess(path)

    sorted_list = label_ent(img_arrays, method, sort=True, ent_norm=ent_norm, color_weight=color_weight)

    print('\nThe images are sorted by entropy.')

    save_img(dst_folder, sorted_list)


def sort_by_noise(path, method, color_weight=None):
    dst_folder = f'../processed/noise_m={method}_t={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    n = 10

    img_arrays, _ = preprocess(path)
    img_arrays = noise_by_increment(img_arrays[0], n)

    sorted_list = label_ent(img_arrays, method, sort=True, color_weight=color_weight)

    print('\nThe images are sorted by entropy.')

    save_img(dst_folder, sorted_list)


def norm_ent():
    path = '../datasets/fixed_noise.bmp'
    img_arr = np.array(Image.open(path))
    N = {}
    for ent in ent_methods:
        N[ent] = calc_ent(img_arr, ent)
        print(f'Entropy calculated for method: {ent}.')

    with open('data/ent_norm.json', 'w') as file:
        json.dump(N, file)


def ent_for_img(path):
    img_arr = np.array(Image.open(path))
    for ent in ent_methods:
        s = calc_ent(img_arr, ent)
        print(f'Entropy: {s},  Method: {ent}')


def main():
    method = 'naive'
    colors = 'hsb'
    # folder_path = '../datasets/satellite/argentina'

    state = 'argentina'
    coo_json = f'../datasets/coordinates/coo_{state}.json'
    folder_path = f"../datasets/satellite/{state}"

    with open('data/ent_norm.json', 'r') as file:
        ent_norm = json.load(file)
    if method not in ent_norm:
        fixed_noise = np.array(Image.open('../datasets/fixed_noise.bmp'))
        ent_norm[method] = calc_ent(fixed_noise, method)
        with open('data/ent_norm.json', 'w') as file:
            json.dump(ent_norm, file)

    folder_path = normalize_path(folder_path)
    print(f'Dataset path: {os.path.abspath(folder_path)}')

    # random_satellite_img(coo_json, 14, save_path=sat_img_path, n_pics=25)
    sort_folder(folder_path, method, colors, ent_norm)
    # sort_by_noise(folder_path, method)
    # ent_for_img(r"C:\scripts\entropy-and-symmetry\datasets\noising\18.png")
    # norm_ent()


if __name__ == '__main__':
    main()
