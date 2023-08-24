from datetime import datetime
import warnings
from functions import *

warnings.filterwarnings("ignore")

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
    'RGBCM'
]

color_opts = ['rgb',
              'hsb',
              'YCbCr',
              'greyscale']

datasets = {'china': "../datasets/satellite/china",
            'usa': "../datasets/satellite/usa",
            'argentina': "../datasets/satellite/argentina",
            'satellite': "../datasets/satellite",
            'noise': "../datasets/noising"}


def sort_folder(path, method, colors='rgb', ent_norm=None, color_weight=None):
    # colors = 'greyscale'
    dst_folder = f'../processed/m={method}_t={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    img_arrays, _ = preprocess(path)

    sorted_list = label_ent(img_arrays, method, sort=True, ent_norm=ent_norm, color_weight=color_weight, colors=colors)

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


def main():
    method = 'RGBCM'
    colors = 'YCbCr'
    eps = 0.2

    folder_path = datasets['argentina']
    sat_img_path = '../datasets/satellite'
    coo = ['-33.1338, -68.7773', '-33.0785, -68.4561']

    ent_norm = get_ent_norm(method)
    print(f'Dataset path: {os.path.abspath(folder_path)}')

    # random_satellite_img(coo_json, 14, save_path=sat_img_path, n_pics=25)
    # for c in coo:
    #     get_google_map_image(c, 14, 500, 500, sat_img_path)
    # sort_folder(folder_path, method, colors, ent_norm, color_weight=(1 - 2 * eps, eps, eps))
    sort_by_noise(datasets['noise'], method)


if __name__ == '__main__':
    main()
