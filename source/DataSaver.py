import os
import json
import platform
import subprocess


class DataSaver:
    def __init__(self, destination, methods):
        self.destination = destination
        self.methods = methods
        self.json_path = os.path.join(self.destination, 'entropy_results.json')
        self.json_file = None

        if not os.path.exists(self.destination):
            os.makedirs(self.destination)
        # Initialize the JSON file if it doesn't exist
        if not os.path.exists(self.json_path):
            with open(self.json_path, 'w') as f:
                f.write("[]")

    def save(self, image):
        if not os.path.exists(self.destination):
            os.makedirs(self.destination)
        srcpath = os.path.relpath(image.path, self.destination)
        newpath = os.path.join(self.destination, os.path.basename(image.path))
        try:
            os.symlink(srcpath, newpath)
        except FileExistsError:
            return

    def save_ent_result(self, images):
        # Create a dictionary to hold the entropy results
        ent_results = []

        # Loop through the images and collect their entropy results
        for index, image in enumerate(images):
            ent_results.append(self.get_ent_result(image))

        # Write the entropy results to the JSON file
        with open(self.json_path, 'w') as f:
            json.dump(ent_results, f, indent=4)

    def auto_save_ent_result(self, images):
        ent_results = []
        with open(self.json_path, 'rb+') as f:
            f.seek(-1, os.SEEK_END)

            # If file is not empty, add a comma
            if f.tell() > 1:
                f.write(b',')

            # Loop through the images and collect their entropy results
            for index, image in enumerate(images):
                ent_results.append(self.get_ent_result(image))

            # Iterate through the ent_results dictionary
            for i, entry in enumerate(ent_results):
                f.write(json.dumps(entry).encode('utf-8'))

                # Add a comma except for the last item
                if i < len(ent_results) - 1:
                    f.write(b',')

            # Add the closing ']'
            f.write(b']')

    def save_single_ent_result(self, index, image):  # Do not use this
        if self.json_file is None:
            self.json_file = open(self.json_path, 'w')
        ent_results = {f"image_{index}.bmp": self.get_ent_result(image)}

        # Write the entropy results to the JSON file

        json_str = json.dumps(ent_results, indent=4)
        self.json_file.write("\n" + json_str)

    def get_ent_result(self, image):
        # Assuming image.entropyResults is a list [value1, value2, ...]
        ent_result_with_method = []
        for i, ent_value in enumerate(image.entropyResults):
            method_name = self.methods[i]
            ent_result_with_method.append({
                "method": method_name,
                "result": ent_value
            })

        label = os.path.basename(os.path.dirname(image.path))
        location = os.path.basename(image.path).split('_')[-2:-4:-1]
        # Add the path field of the Image object
        return {
            "index": image.index,
            "path": image.path,
            "size": image.size,
            "pixel size": image.pixel_size,
            "location": location,
            "label": label,
            "entropy_results": ent_result_with_method
        }

    def prettify_json_file(self):
        try:
            with open(self.json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print("JSON Decode Error")
            return

        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=4)
