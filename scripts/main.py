from datetime import datetime
import warnings

from functions import *

warnings.filterwarnings("ignore")

ent_methods = [
    'hist',
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
            'satellite': "../datasets/satellite"}


def sort_folder(path, method, colors='rgb', ent_norm=None, color_weight=None, linearCombine=False,
                method_weight=None):
    # colors = 'greyscale'
    if isinstance(method, str):
        m_name = method
    elif isinstance(method, list):
        m_name = '-'.join(method)
    elif isinstance(method, dict):
        m_name = '-'.join(method.keys())
    else:
        return

    dst_folder = f'../processed/m={m_name}_t={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    img_arrays, _ = preprocess(path)
    if linearCombine:
        sorted_list = linearCombine_ent(img_arrays, method, method_weight, sort=True, ent_norm=ent_norm, colors=colors,
                                        color_weight=color_weight)
    else:
        sorted_list = label_ent(img_arrays, method, sort=True, ent_norm=ent_norm, colors=colors,
                                color_weight=color_weight)

    print('\nThe images are sorted by entropy.')

    save_img(dst_folder, sorted_list)


def sort_by_noise(path, method, color_weight=None, colors='rgb', ent_norm=None, linearCombine=False,
                  method_weight=None):
    dst_folder = f'../processed/noise_m={method}_t={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    n = 10

    img_arrays, _ = preprocess(path)
    img_arrays = noise_by_increment(img_arrays[0], n)
    if linearCombine:
        sorted_list = linearCombine_ent(img_arrays, ent_methods, method_weight, sort=True, ent_norm=ent_norm,
                                        colors=colors, color_weight=color_weight)
    else:
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


def check_norm():
    path = '../datasets/fixed_noise.bmp'
    ent_norm = get_ent_norm()
    ent_for_img(path, ent_methods, ent_norm)


def sort_images():
    method = 'lbp'
    colors = 'YCbCr'
    path = '../datasets/satellite/more'
    dst_folder = f'../processed/m={method}_t={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    img_arrays, paths = preprocess(path)
    img_arrays = label_ent(img_arrays, method, sort=False, colors=colors)

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    for i in range(len(paths)):
        file_path = os.path.join(dst_folder, f"s={img_arrays[i][1]}_{os.path.basename(paths[i])}")
        Image.fromarray(img_arrays[i][0], 'RGB').save(file_path)

    os.startfile(os.path.join(os.getcwd(), dst_folder))


def main():
    # method = {'hist': 1 / 2, 'naive': 1 / 2}
    colors = 'YCbCr'
    eps = 0.2

    ent_norm = get_ent_norm()

    folder_path = datasets['satellite']
    sat_img_path = '../datasets/satellite/agriculture'
    coo = ['-33.1338, -68.7773', '-33.0785, -68.4561']

    print(f'Dataset path: {os.path.abspath(folder_path)}')
    coo_json = '../datasets/coordinates/coo_africa.json'

    # check_norm()
    # random_satellite_img(coo_json, 14, save_path=sat_img_path, n_pics=100)
    # for c in coo:
    #     get_google_map_image(c, 14, 500, 500, sat_img_path)
    # for method in ent_methods:
    #     sort_folder(folder_path, method, colors, ent_norm=ent_norm, color_weight=(1 - 2 * eps, eps, eps),
    #                 linearCombine=False, method_weight=None)


if __name__ == '__main__':
    main()
