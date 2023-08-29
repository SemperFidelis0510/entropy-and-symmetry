import os
import hashlib
from shutil import copyfile
from PIL import Image
import numpy as np


def calculate_sha256(image_path):
    """Calculate SHA-256 hash of the first 100 RGB pixels of an image."""
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    # Flatten the first 100 RGB pixels into a 1D array
    flat_pixels = img_array[:10, :10, :3].flatten()

    # Convert to bytes
    pixel_bytes = flat_pixels.tobytes()

    # Calculate SHA-256 hash
    sha256_hash = hashlib.sha256()
    sha256_hash.update(pixel_bytes)
    return sha256_hash.hexdigest()


# Path to the datasets folder and source folder
datasets_path = "../datasets/classified_pictures"
image_directory = "../datasets/satellite"

# Create a dictionary to store SHA-256 hashes of source files
source_hashes = {}
image_paths = []
if os.path.isdir(image_directory):
    for root, _, filenames in os.walk(image_directory):
        for filename in filenames:
            if filename.lower().endswith(('png')):
                img_path = os.path.join(root, filename)
                image_paths.append(img_path)
# Calculate SHA-256 hashes for all files in the source folder
for path in image_paths:
    file_hash = calculate_sha256(path)
    print(f"Calculated hash for {path}: {file_hash}")  # Debugging line
    source_hashes[file_hash] = path

# Loop through each subfolder in the datasets folder
for subfolder in os.listdir(datasets_path):
    subfolder_path = os.path.join(datasets_path, subfolder)

    # Make sure it's a folder
    if not os.path.isdir(subfolder_path):
        continue

    # Loop through each file in the subfolder
    for filename in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, filename)

        # Make sure it's a file
        if not os.path.isfile(file_path):
            continue

        # Calculate the SHA-256 hash of the file
        file_hash = calculate_sha256(file_path)
        print(f"Looking up hash for {file_path}: {file_hash}")  # Debugging line

        # Check if this hash exists in the source folder
        if file_hash in source_hashes:
            original_name = source_hashes[file_hash]  # Debugging line
            new_file_name = os.path.basename(original_name)
            # Construct the new path for the file
            new_file_path = os.path.join(subfolder_path, new_file_name)
            print(f"Match found for {file_path}. Renaming to {new_file_path}.")
            # Rename the file
            os.rename(file_path, new_file_path)
