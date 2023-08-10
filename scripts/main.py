from functions import *


def sort_folder(path):
    method = 'hist'
    dst_folder = f'../processed/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    list_of_paths = load_images(path)
    img_list = preprocess(list_of_paths)

    sorted_list = sorted(img_list, key=lambda x: calc_ent(x, method))

    save_img(dst_folder, sorted_list)


def main():
    folder_path = r'C:\scripts\entropy-and-symmetry\datasets\pattern_images'
    sort_folder(folder_path)


if __name__ == '__main__':
    main()
