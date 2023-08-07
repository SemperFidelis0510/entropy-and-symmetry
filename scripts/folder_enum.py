import os
import sys
from PIL import Image


def convert_and_number_files_in_folder(path):
    # Convert .jpg and .webp files to .png
    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.webp'):
            img_path = os.path.join(path, filename)
            img = Image.open(img_path)
            new_name = os.path.splitext(filename)[0] + '.png'
            img.save(os.path.join(path, new_name))
            os.remove(img_path)

    # Number all files in the folder
    for idx, filename in enumerate(sorted(os.listdir(path))):
        file_extension = os.path.splitext(filename)[1]
        new_name = f"{idx}{file_extension}"
        os.rename(os.path.join(path, filename), os.path.join(path, new_name))


if len(sys.argv) < 2:
    print("Please provide the folder path as an argument.")
    sys.exit(1)

folder_path = sys.argv[1]
convert_and_number_files_in_folder(folder_path)
print("Files have been converted to .png and numbered successfully.")
