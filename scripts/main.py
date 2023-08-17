from datetime import datetime

from functions import *


def sort_folder(path):
    # colors = 'greyscale'
    colors = 'rgb'
    method = 'dwt'

    dst_folder = f'../processed/m={method}_t={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    img_arrays, _ = preprocess(path, colors=colors)

    img_arrays = label_ent(img_arrays, method, sort=True)

    print('\nThe images are sorted by entropy.')

    save_img(dst_folder, img_arrays)


def main():
    # folder_path = r'../datasets/pattern_images'
    #folder_path = r'../datasets/web_symmetries/rotation'
    folder_path = r'../datasets/pattern_images'
    folder_path = normalize_path(folder_path)
    print(f'Dataset path: {os.path.abspath(folder_path)}')
    sort_folder(folder_path)


if __name__ == '__main__':
    main()
