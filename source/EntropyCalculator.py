import numpy as np
from source.Image import Image
import json
import torch

class EntropyCalculator:
    def __init__(self, color_weight = None, ent_norm_path = None, reset_norm = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.color_weight_gpu = torch.tensor(
            [0.299, 0.587, 0.114]).to(self.device)  # Assuming you want to use the same weights and move them to GPU
        self.color_weight = color_weight or (0.2989, 0.5870, 0.1140)
        self.ent_norm_path = ent_norm_path or '../source/data/entropy_results.json'
        self.reset_norm = reset_norm
        self.ent_norm = self.get_ent_norm()

    def calculateEntropy(self, image: Image):
        for method, processedData in image.processedData.items():
            if method == 'adapt':
                temp = []
                for level, matrix in enumerate(processedData):
                    ent_matrix = []
                    for row_index, row in enumerate(matrix):
                        ent_row = []
                        for column_index, sub_image in enumerate(row):
                            segment_entropies = []
                            for segment in sub_image:
                                segment_entropies.append(self.entropy(segment, self.get_norm(method, level, row_index, column_index)))
                            ent_row.append(np.mean(segment_entropies))
                        ent_matrix.append(ent_row)
                    temp.append(ent_matrix)
            elif method == 'dwt':
                ent = []
                for level in range(len(processedData[0])):
                    norm = self.get_norm(method, level)
                    result = 0
                    for color in processedData:
                        if level == 0: #approximation coefficient
                            data = color[level].flatten()
                        else:
                            data = color[1][level].flatten()
                        result += self.entropy_gpu(data, norm)
                    ent.append(result)
                temp = ent
            else:
                temp = []
                for level, matrix in enumerate(processedData):
                    ent_matrix = []
                    for row_index, row in enumerate(matrix):
                        ent_row = []
                        for column_index, sub_image in enumerate(row):
                          ent_row.append(self.entropy_gpu(sub_image, self.get_norm(method, level, row_index, column_index)))
                        ent_matrix.append(ent_row)
                    temp.append(ent_matrix)
            image.entropyResults.append(temp)

    def get_norm(self, method, level, row=None, column=None):
        if self.reset_norm:
            return 1
        if row is not None:
            norm = self.ent_norm[method][level][row][column]
        else: # dwt
            norm = self.ent_norm[method][level]
        return norm

    def entropy_gpu(self, Data, norm=1):
        Data = Data.abs()
        ent = 0
        if Data.dim() == 3:
            if Data.size(-1) == 3:  # Check if the last dimension has 3 channels (RGB)
                Data = torch.matmul(Data, self.color_weight_gpu.to(Data.dtype))

        total_sum = torch.sum(Data)
        if total_sum == 0:
            return 0

        normalize_arr = Data / total_sum
        ent = -torch.sum(normalize_arr * torch.log2(normalize_arr + torch.finfo(torch.float32).eps))
        ent = ent/norm
        ent = ent.cpu()
        return ent.item()

    def entropy(self, Data, norm=1):
        arr = np.abs(Data)
        ent = 0
        if arr.ndim == 3:
            if arr.shape[-1] == 3:  # Check if the last dimension has 3 channels (RGB)
                arr = np.dot(arr, self.color_weight)
        total_sum = np.sum(arr)
        if total_sum == 0:
            return 0
        normalize_arr = arr / total_sum
        ent = -np.sum(normalize_arr * np.log2(normalize_arr + np.finfo(float).eps))
        return ent / norm

    def get_ent_norm(self):
        if self.reset_norm:
            return
        # Read the JSON file
        with open(self.ent_norm_path, 'r') as f:
            data = json.load(f)

        # Navigate through the JSON structure to get the 'entropy_results' list
        entropy_results = data[0].get("entropy_results", [])

        all_results = {}

        # Loop through the 'entropy_results' list to collect all methods and their results
        for item in entropy_results:
            method = item.get("method")
            result = item.get("result")
            all_results[method] = result
        return all_results
