import os
import random
import json
import geopandas as gpd
from shapely.geometry import Point
from functions import get_google_map_image
from PIL import Image

# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 使用os.path.join()将脚本的路径与相对路径结合起来，从而形成绝对路径
path_to_china_boundary = os.path.join(script_dir, "../resources/gadm36_CHN_shp/gadm36_CHN_0.shp")
gdf = gpd.read_file(path_to_china_boundary)

# 获取中国的边界
china_boundary = gdf.geometry.iloc[0]

# 中国的大致经纬度范围（用于生成随机点，然后检查是否在边界内）
LAT_MIN, LAT_MAX = 18.0, 53.5
LON_MIN, LON_MAX = 73.5, 135.0

# 定义缩放级别
zoom_levels = [14]

# 确保保存目录存在
save_directory = os.path.join(script_dir, "../datasets/satellite/")
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# JSON文件的保存路径
json_save_path = os.path.join(script_dir, "../datasets/satellite/coord_names.json")

# 在中国的边界内随机选择点
num_images = 10
coord_names = []
for _ in range(num_images):
    coords_for_this_run = []
    while True:
        latitude = random.uniform(LAT_MIN, LAT_MAX)
        longitude = random.uniform(LON_MIN, LON_MAX)
        location_point = Point(longitude, latitude)
        if china_boundary.contains(location_point):
            break

    location = f"{latitude},{longitude}"

    for zoom in zoom_levels:
        image_array = get_google_map_image(location, zoom_level=zoom, save=False)
        
        # 保存图像
        filename = f"{save_directory}map_image_{location.replace(',', '_')}_{zoom}.png"
        image = Image.fromarray(image_array)
        image.save(filename)
        
    coords_for_this_run.append(location)
    coord_names.append(coords_for_this_run)

# 保存坐标名到JSON文件
with open(json_save_path, 'w') as json_file:
    json.dump(coord_names, json_file)



