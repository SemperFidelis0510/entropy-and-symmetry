import os
import json
import logging
from source.Logger import Logger
import torch
class SystemInitializer(Logger):
    def __init__(self, src_folder, dst_folder, image_format=('png', 'bmp'),
                 head=None, single_batch_size=300, max_queue_size=30, preprocess_channels='rgb',
                 ent_norm_path=None):
        self.src_folder = src_folder
        self.dst_folder = dst_folder
        self.all_data_paths = []
        self.already_processed_paths = []
        self.need_to_process_paths = []
        self.image_format = image_format
        self.head = head
        self.json_path = os.path.join(self.dst_folder, 'entropy_results.json')
        self.total_batch = 0
        self.single_batch_size = single_batch_size
        self.max_queue_size = max_queue_size
        self.set_logger()
        self.preprocessed_channels = preprocess_channels
        self.ent_norm_path = ent_norm_path or '../source/data/entropy_results.json'
        self.run_device = "gpu" if torch.cuda.is_available() else "cpu"
    def initSystemState(self):
        self.get_all_data_paths()
        self.get_already_processed_paths()
        self.need_to_process_paths = list(set(self.all_data_paths) - set(self.already_processed_paths))
        if self.head is not None:
            self.need_to_process_paths = self.need_to_process_paths[:self.head]
        self.get_total_batch()
        self.printSystemState()
        self.logSystemState()

    def set_logger(self):
        logger = logging.getLogger("process_logger")
        # Create a FileHandler and Formatter, then add the handler to the logger
        if not os.path.exists(self.dst_folder):
            os.makedirs(self.dst_folder)
        file_handler = logging.FileHandler(os.path.join(self.dst_folder, "application.log"))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    def get_total_batch(self):
        size = len(self.need_to_process_paths)
        self.total_batch = size//self.single_batch_size
        if size%self.single_batch_size:
            self.total_batch += 1
    def get_already_processed_paths(self):
        if not os.path.exists(self.dst_folder):
            os.makedirs(self.dst_folder)
            return
        if not os.path.exists(self.json_path):
            return
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        for record in data:
                if 'path' in record:
                    self.already_processed_paths.append(record['path'])
    def get_all_data_paths(self):
        if os.path.isdir(self.src_folder):
            for root, _, filenames in os.walk(self.src_folder):
                for filename in filenames:
                    if filename.lower().endswith(self.image_format):
                        img_path = os.path.join(root, filename)
                        self.all_data_paths.append(img_path)
        elif os.path.isfile(self.src_folder) and self.src_folder.lower().endswith(self.image_format):
            self.all_data_paths = [self.src_folder]
        else:
            raise ValueError("The provided path is neither a directory nor a valid image file.")
    def printSystemState(self):
        print('System State')
        print(f' * Image Data Path: {self.src_folder}')
        print(f' * Total Image: {len(self.all_data_paths)}')
        print(f' * Already Process: {len(self.already_processed_paths)}')
        print(f' * Image To Process: {len(self.need_to_process_paths)}')
        print(f' * Total Batch: {self.total_batch}')
        print(f' * Single Batch Size: {self.single_batch_size}')
        print(f' * Auto Save: for every {self.max_queue_size}')
        print(f' * Run on: {self.run_device}')
        print(f' * Entropy norm path: {self.ent_norm_path}')
        print(f' * Output folder: {self.dst_folder}')

    def logSystemState(self):
        self.log_message('System State')
        self.log_message(f' * Image Data Path: {self.src_folder}')
        self.log_message(f' * Total Image: {len(self.all_data_paths)}')
        self.log_message(f' * Already Process: {len(self.already_processed_paths)}')
        self.log_message(f' * Image To Process: {len(self.need_to_process_paths)}')
        self.log_message(f' * Total Batch: {self.total_batch}')
        self.log_message(f' * Single Batch Size: {self.single_batch_size}')
        self.log_message(f' * Auto Save: for every {self.max_queue_size}')
        self.log_message(f' * Run on: {self.run_device}')
        self.log_message(f' * Entropy norm path: {self.ent_norm_path}')
        self.log_message(f' * Output folder: {self.dst_folder}')



if __name__ == '__main__':
    print('Test systeminitializer class')
    src_folder = "../datasets/classified_pictures"
    dst_folder = "../tests/data/test_systeminitializer"
    systemIni = SystemInitializer(src_folder=src_folder, dst_folder=dst_folder, head=10)
    systemIni.initSystemState()