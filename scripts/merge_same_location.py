import json
import os

def merge_labels_based_on_location(data_list):
    location_to_entry = {}

    for entry in data_list:
        loc_key = tuple(entry['location'])

        if loc_key not in location_to_entry:
            location_to_entry[loc_key] = entry
        else:
            if not isinstance(location_to_entry[loc_key]['label'], list):
                location_to_entry[loc_key]['label'] = [location_to_entry[loc_key]['label']]
            location_to_entry[loc_key]['label'].append(entry['label'])

    # Return data list with merged labels and without duplicates
    return list(location_to_entry.values())

# Load the JSON data
with open('/Users/tangjingqin/Desktop/entropy-and-symmetry/processed/results/entropy_results.json', 'r') as file:
    data_list = json.load(file)

merged_data_list = merge_labels_based_on_location(data_list)

# Ensure the new folder exists or create it
# Save the modified data to a file in the new folder with proper formatting

with open('/Users/tangjingqin/Desktop/entropy-and-symmetry/processed/results/123.json', 'w') as file:
    json.dump(merged_data_list, file, indent=4, separators=(',', ': '))
