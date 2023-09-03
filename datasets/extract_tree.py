import os
import json
def write_folder_structure_to_json(root_folder, json_file_path):
    folder_structure = {}

    for subdir, dirs, files in os.walk(root_folder):
        relative_path = os.path.relpath(subdir, root_folder)
        folder_structure[relative_path] = files

    with open(json_file_path, 'w') as f:
        json.dump(folder_structure, f, indent=4)
write_folder_structure_to_json('../datasets/classified_pictures', 'tree.json')