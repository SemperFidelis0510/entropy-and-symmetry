import os
import random
import json
import geopandas as gpd
from shapely.geometry import Point
from functions import get_google_map_image
from PIL import Image

# Obtain the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Use os. path. coin() to combine the path of the script with the relative path to form an absolute path
path_to_china_boundary = os.path.join(script_dir, "../resources/gadm36_CHN_shp/gadm36_CHN_0.shp")
gdf = gpd.read_file(path_to_china_boundary)

# Obtain China's borders
china_boundary = gdf.geometry.iloc[0]

# The approximate latitude and longitude range of China (used to generate random points and check if they are within the boundary)
LAT_MIN, LAT_MAX = 18.0, 53.5
LON_MIN, LON_MAX = 73.5, 135.0

# Define Zoom Level
zoom_levels = [14]

# Ensure that the save directory exists
save_directory = os.path.join(script_dir, "../datasets/satellite/")
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Save path for JSON files
json_save_path = os.path.join(script_dir, "../datasets/satellite/coord_names.json")

# Randomly select points within the boundaries of China
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

        # Save image
        filename = f"{save_directory}map_image_{location.replace(',', '_')}_{zoom}.png"
        image = Image.fromarray(image_array)
        image.save(filename)

    coords_for_this_run.append(location)
    coord_names.append(coords_for_this_run)

# Save coordinate names to JSON files
with open(json_save_path, 'w') as json_file:
    json.dump(coord_names, json_file)
