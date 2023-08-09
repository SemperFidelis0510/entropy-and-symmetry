from functions import *


def transform(img):
    pass


def main():
    folder_path = ''
    method = 'hist'

    list_of_paths = load_images(folder_path)
    img_list = []

    for img_path in list_of_paths:
        img = preprocess(img_path)
        # img = transform(img)
        s = calc_ent(img, method)
        img_list.append([img, s])

    sorted_list = sorted(img_list, key=lambda x: x[1])


if __name__ == '__main__':
    main()
