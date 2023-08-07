#! /bin/python3
import os

def number_files_in_folder(folder_path):
    for idx, filename in enumerate(sorted(os.listdir(folder_path))):
        file_extension = os.path.splitext(filename)[1]
        new_name = f"{idx}{file_extension}"
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))

folder_path = input('folder path please: ')
number_files_in_folder(folder_path)
print("Files have been numbered successfully.")
