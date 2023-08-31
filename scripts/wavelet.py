from entropy import *
from transforms import *


def test_dwt(image, flag=0):
    r_dwt = dwt(image[:, :, 0])
    g_dwt = dwt(image[:, :, 1])
    b_dwt = dwt(image[:, :, 2])
    transform = [r_dwt, g_dwt, b_dwt]
    result = dict()
    sum_ent = 0
    if flag == 0:
        for level in range(len(r_dwt)):
            for color in transform:
                color_ent = 0
                if level == 0:
                    for i in range(2):  # sum up three direction
                        color_ent += entropy(np.array(color[level][i]))
                else:
                    for i in range(3):  # sum up three direction
                        color_ent += entropy(color[level][i])
                result[level] = result.get(level, 0) + color_ent
    else:
        for level in range(len(r_dwt)):
            for color in transform:
                color_ent = 0
                if level == 0:
                    for i in range(2):  # sum up three direction
                        color_ent += entropy(np.array(color[level][i]))
                else:
                    arr = np.array(color[level]).flatten()
                    color_ent = entropy(arr)
                result[level] = result.get(level, 0) + color_ent
    return result


from PIL import Image
from functions import *

img_paths = '/home/yanglin/Study/2023B/Technion_Summer/entropy-and-symmetry/datasets/Fractals with controlled parameter/Shape-Replacement Fractals/low'
imgs_arr, paths = preprocess(img_paths)
paths = [file.split('/')[len(file.split('/')) - 1] for file in paths]  # get short file name
result = []
n = len(imgs_arr)
start_time = time.time()
print(f'\rComputing dwt: {print_progress_bar(0, n, start_time=start_time)}', end='', flush=True)
for i, img in enumerate(imgs_arr):
    d = test_dwt(img, 1)
    d['file'] = paths[i]
    result.append(d)
    print(f'\rComputing dwt: {print_progress_bar(i + 1, n, start_time=start_time)}', end='', flush=True)
print()
sorted_a = sorted(result, key=lambda d: d[0])
for x in sorted_a:
    print(x)
