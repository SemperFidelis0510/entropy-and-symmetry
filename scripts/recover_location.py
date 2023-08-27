import os
import hashlib
from shutil import copyfile


def calculate_sha256(file_path):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# Path to the datasets folder and source folder
datasets_path = "../datasets/classified_pictures"
source_path = "path/to/source/folder"

# Create a dictionary to store SHA-256 hashes of source files
source_hashes = {}

# Calculate SHA-256 hashes for all files in the source folder
for filename in os.listdir(source_path):
    file_path = os.path.join(source_path, filename)
    if os.path.isfile(file_path):
        file_hash = calculate_sha256(file_path)
        source_hashes[file_hash] = filename

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

        # Check if this hash exists in the source folder
        if file_hash in source_hashes:
            original_name = source_hashes[file_hash]

            # Construct the new path for the file
            new_file_path = os.path.join(subfolder_path, original_name)

            # Rename the file
            os.rename(file_path, new_file_path)

