from io import BytesIO

import numpy as np
import requests
from PIL import Image


def get_google_map_image(latitude, longitude, zoom_level, width, height, api_key):
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{latitude},{longitude}",
        "zoom": zoom_level,
        "size": f"{width}x{height}",
        "maptype": "satellite",
        "key": api_key
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        raise Exception(f"Error retrieving image: {response.text}")


# Example usage
latitude = 40.702147
longitude = -74.015794
zoom_level = 13
width = 600
height = 300
api_key = "AIzaSyAnFhz63LUhs9BGZfU_MW5EATc-s9r7epQ"

image = get_google_map_image(latitude, longitude, zoom_level, width, height, api_key)
img = np.array(image)
img = img[:-30, :]
image.save("map_image.png")
