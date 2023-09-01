from PIL import Image
import os

# Define the path to the main folder
main_folder_path = r"C:\Users\obser\PycharmProjects\entropy-and-symmetry\datasets\classified_pictures"

# Loop through each subfolder in the main folder
for subfolder in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder)

    # Check if the path is a directory (i.e., a subfolder)
    if os.path.isdir(subfolder_path):

        # Loop through each file in the subfolder
        for filename in os.listdir(subfolder_path):

            # Check if the file is a PNG image
            if filename.endswith(".png"):
                image_path = os.path.join(subfolder_path, filename)

                # Open the image using PIL
                with Image.open(image_path) as img:

                    # Check if the image resolution is 478x478
                    if img.size == (478, 478):
                        # Resize the image to 1000x978
                        img_resized = img.resize((1000, 978), 3)

                        # Save the resized image back to the same location
                        img_resized.save(image_path)

print("Image resizing completed.")
