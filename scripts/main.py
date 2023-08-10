from datetime import datetime

from functions import *


def main():
    method = 'hist'
    folder_path = r'C:\scripts\entropy-and-symmetry\datasets\pattern_images'
    dst_folder = f'../processed/{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    list_of_paths = load_images(folder_path)
    img_list = preprocess(list_of_paths)

    sorted_list = sorted(img_list, key=lambda x: calc_ent(x, method))

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    i = 0
    for obj in sorted_list:
        save_img(os.path.join(dst_folder, f'{i}.bmp'), obj)
        i += 1


if __name__ == '__main__':
    main()
