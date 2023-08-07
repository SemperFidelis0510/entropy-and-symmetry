import os
import sys
from PIL import Image

# Chane working folder to parent folder "entropy-and-symmetry"
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert_and_number_files_in_folder(path):
    # Convert .jpg, .jpeg, and .webp files to .png
    for filename in os.listdir(path):
        if filename.endswith(('.jpg', '.jpeg', '.webp', '.gif')):
            img_path = os.path.join(path, filename)
            img = Image.open(img_path)
            new_name = os.path.splitext(filename)[0] + '.png'
            img.save(os.path.join(path, new_name))
            os.remove(img_path)

    # Temporary mapping of new names
    new_names_mapping = {}
    for idx, filename in enumerate(sorted(os.listdir(path))):
        file_extension = os.path.splitext(filename)[1]
        new_name = f"{idx}{file_extension}"
        new_names_mapping[filename] = new_name

    # Rename files using temporary mapping to avoid name clashes
    for old_name, new_name in new_names_mapping.items():
        temp_name = "temp_" + new_name
        os.rename(os.path.join(path, old_name), os.path.join(path, temp_name))

    # Remove temporary prefix
    for filename in os.listdir(path):
        if filename.startswith("temp_"):
            os.rename(os.path.join(path, filename), os.path.join(path, filename[5:]))


if len(sys.argv) < 2:
    print("Please provide the folder path as an argument.")
    sys.exit(1)

folder_path = sys.argv[1]
convert_and_number_files_in_folder(folder_path)
print("Files have been converted to .png and numbered successfully.")
