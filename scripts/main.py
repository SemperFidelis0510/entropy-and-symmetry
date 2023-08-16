from datetime import datetime

from functions import *


def sort_folder(path):
    # colors = 'greyscale'
    colors = 'rgb'
    method = 'dft'

    dst_folder = f'../processed/m={method}_t={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    img_arrays = preprocess(path, colors=colors)

    img_arrays = label_ent(img_arrays, method)
    sorted_list = sorted(img_arrays, key=lambda x: x[1])

    print('\nThe images are sorted by entropy.')

    save_img(dst_folder, sorted_list)


def main():
    # folder_path = r'../datasets/pattern_images'
    folder_path = r'../datasets/web_symmetries/rotation'
    # folder_path = r'../datasets/Fractals with controlled parameter/Line-Replacement Fractals/snowflake'
    folder_path = normalize_path(folder_path)
    print(f'Dataset path: {os.path.abspath(folder_path)}')
    sort_folder(folder_path)


if __name__ == '__main__':
    main()
