import os
import sys
import json
import numpy as np
from PIL import Image
from scripts.utils import print_progress_bar


def cut(ds, n):
    return ds[:n]


def join_labels(ds):
    old_labels = ['plain nature', 'detailed nature', 'Agriculture', 'villages', 'city']
    new_labels = ['nature', 'country', 'city']
    new_ds = []

    for obj in ds:
        label = obj["label"]
        if isinstance(label, str):
            label = [label]

        if 'city' in label:
            new_label = 'city'
        elif ('Agriculture' in label) or ('villages' in label) or ('country' in label):
            new_label = 'country'
        elif ('plain nature' in label) or ('detailed nature' in label) or ('nature' in label):
            new_label = 'nature'
        else:
            return 1

        # Update the label in the dataset object
        obj['label'] = new_label

        # Update the file path
        old_path = obj["path"]
        folder, filename = os.path.split(old_path)
        parent_folder = os.path.dirname(folder)
        new_folder = os.path.join(parent_folder, new_label)
        new_path = os.path.join(new_folder, filename)

        # Update the path in the dataset object
        obj["path"] = new_path

        new_ds.append(obj)

    return new_ds


def augment_data(ds):
    augmented_dataset = []
    total_images = len(ds)

    for i, entry in enumerate(ds):
        original_path = entry["path"]
        original_label = entry["label"]
        original_entropy_results = entry["entropy_results"]

        # Update the path to match the new label folder
        folder, filename = os.path.split(original_path)
        parent_folder = os.path.dirname(folder)
        new_folder = os.path.join(parent_folder, original_label)
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        updated_path = os.path.join(new_folder, filename)

        # Load the original image
        try:
            img = Image.open(updated_path).convert('RGB')
        except FileNotFoundError:
            print(f"\nFile not found: {updated_path}")
            continue

        # Generate augmented images
        for angle in [0, 90, 180, 270]:
            for flip in [False, True]:
                # Rotate and flip
                new_img = img.rotate(angle)
                if flip:
                    new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)

                # Generate new path
                new_path = os.path.join(
                    new_folder,
                    f"{os.path.splitext(filename)[0]}_rot{angle}_flip{int(flip)}.png"
                )

                # Save the new image, overwrite if exists
                new_img.save(new_path, overwrite=True)

                # Create new dataset entry
                new_entry = {
                    "index": len(augmented_dataset),
                    "path": new_path,
                    "size": os.path.getsize(new_path),
                    "pixel size": new_img.size,
                    "location": entry["location"],
                    "label": original_label,
                    "entropy_results": original_entropy_results  # Entropy results are the same as the original image
                }

                # Add to the augmented dataset
                augmented_dataset.append(new_entry)

        print_progress_bar("Images processed", i + 1, total_images)

    return augmented_dataset


def main():
    path = "../processed/results/entropy_results_joined_labels.json"
    with open(path, 'r') as f:
        dataset = json.load(f)

    # dataset = join_labels(dataset)
    dataset = augment_data(dataset)

    path = "../datasets/classified_pictures/entropy_results_augmented.json"
    with open(path, 'w') as f:
        json.dump(dataset, f, indent=4)


if __name__ == '__main__':
    main()
