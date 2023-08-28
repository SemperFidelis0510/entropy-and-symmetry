import os
import  json
import platform
import subprocess
class DataSaver:
    def __init__(self, destination, methods):
        self.destination = destination
        self.methods = methods
        self.json_path = os.path.join(self.destination, 'entropy_results.json')
        self.json_file = None
    def save(self, index, image):
        if not os.path.exists(self.destination):
            os.makedirs(self.destination)
        path = os.path.join(self.destination, f'image_{index}.bmp')
        image.rawData.save(path)

    def save_ent_result(self, images):
        # Create a dictionary to hold the entropy results
        ent_results = {}

        # Loop through the images and collect their entropy results
        for index, image in enumerate(images):
            ent_results[f"image_{index}.bmp"] = self.get_ent_result(index, image)

        # Write the entropy results to the JSON file
        with open(self.json_path, 'w') as f:
            json.dump(ent_results, f, indent=4)
    def save_single_ent_result(self, index, image):
        if self.json_file is None:
            self.json_file = open(self.json_path, 'w')
        ent_results = {}
        ent_results[f"image_{index}.bmp"] = self.get_ent_result(index, image)

        # Write the entropy results to the JSON file

        json_str = json.dumps(ent_results, indent=4)
        self.json_file.write("\n" + json_str)

    def get_ent_result(self, index, image):
        # Assuming image.entropyResults is a list [value1, value2, ...]
        ent_result_with_method = []
        for i, ent_value in enumerate(image.entropyResults):
            method_name = self.methods[i]
            ent_result_with_method.append({
                "method": method_name,
                "result": ent_value
            })

        label = os.path.basename(os.path.dirname(image.path))

        # Add the path field of the Image object
        return {
            "path": image.path,
            "size": image.size,
            "pixel size": image.pixel_size,
            "label": label,
            "entropy_results": ent_result_with_method
        }