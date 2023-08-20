# get_image.py
import os
import random
from functions import get_google_map_image

LATITUDE_RANGE = (-85, 85)
LONGITUDE_RANGE = (-180, 180)

# Ensure directory exists
save_directory = "../datasets/satellite/"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

for _ in range(10):  # Get 10 images
    lat = random.uniform(*LATITUDE_RANGE)
    lon = random.uniform(*LONGITUDE_RANGE)
    location = f"{lat:.6f},{lon:.6f}"
    image_array = get_google_map_image(location, zoom_level=5, save=True)
