import gc
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
from source.utils import print_progress_bar, open_folder
from tqdm import tqdm
from source.ImageLoader import ImageLoader
from source.Logger import Logger
class PipelineManager(Logger):
    def __init__(self, systemInitializer, preprocessor, processor, entropyCalculator, dataSaver):
        self.systemInitializer = systemInitializer
        self.preprocessor = preprocessor
        self.processor = processor
        self.entropyCalculator = entropyCalculator
        self.dataSaver = dataSaver


    def process_single_image(self, image):
        self.log_message(f'process {image.path}')
        self.preprocessor.applyPreprocessing(image)
        del image.rawData
        self.processor.applyProcessing(image)
        del image.preprocessedData
        self.entropyCalculator.calculateEntropy(image)
        del image.processedData
        return image

    def run_single_batch_process(self, curr_batch, batch_size, total_batch, images_path, max_queue_size, save_queue):
        start = curr_batch * batch_size
        end = (curr_batch + 1) * batch_size
        base_index = len(self.systemInitializer.already_processed_paths) + start
        print(f'Batch Process ({curr_batch}/{total_batch}) {start+1}-{end}')
        self.log_message(f'Batch Process ({curr_batch}/{total_batch}) {start+1}-{end}')
        image_objects = ImageLoader.load_images(images_path[start:end], base_index)
        n = len(image_objects)
        start_time = time.time()
        for index, image_object in enumerate(image_objects):
            self.process_single_image(image_object)
            save_queue.append(image_object)
            if len(save_queue) == max_queue_size:
                for queue_index, image_object in enumerate(save_queue):
                    self.dataSaver.save(image_object)
                self.dataSaver.auto_save_ent_result(save_queue)
                save_queue.clear()
            print_progress_bar('Entropy calculation', index + 1, n, start_time=start_time)
        self.dataSaver.prettify_json_file()
        print(f'Batch Process {curr_batch} Done.')
        self.log_message(f'Batch Process {curr_batch} Done.')
    def runPipeline(self):
        self.log_message("program start")
        self.systemInitializer.initSystemState()
        images_path = self.systemInitializer.need_to_process_paths
        total_batch = self.systemInitializer.total_batch
        batch_size = self.systemInitializer.single_batch_size
        max_queue_size = self.systemInitializer.max_queue_size
        save_queue = []
        for curr_batch in range(total_batch):
            self.run_single_batch_process(curr_batch, batch_size, total_batch,
                                          images_path, max_queue_size, save_queue)
        # Saving result
        size = len(save_queue)
        if size:  # last queue
            start_time = time.time()
            for queue_index, image_object in enumerate(save_queue):
                self.dataSaver.save(image_object)
                print_progress_bar('Saving result in the queue', queue_index + 1, size, start_time=start_time)
            self.dataSaver.auto_save_ent_result(save_queue)
        self.dataSaver.prettify_json_file()

        open_folder(self.dataSaver.destination)

    def process_and_save_single_image(self, image, index):
        # Step 1: Process the image
        self.process_single_image(image)

        # Step 2: Save the processed image
        self.dataSaver.save(index, image)

        # Step 3: Save the entropy result
        self.dataSaver.save_single_ent_result(index, image)
        del image
        gc.collect()

from source.Preprocessor import Preprocessor
from source.Processor import Processor
from source.EntropyCalculator import EntropyCalculator
from source.DataSaver import DataSaver
from source.SystemInitializer import SystemInitializer
import warnings
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    print('Test systeminitializer class')
    src_folder = "../datasets/classified_pictures"
    dst_folder = "../tests/data/test_pipelineManager"
    process_methods_with_params = {'laplace': None, 'joint_red_green': None, 'joint_all': None,
                               'lbp': None, 'lbp_gabor': None, 'RGBCM': None,
                               'dft': None, 'naive': None, 'dwt': {'wavelet': 'haar', 'level': 'all'}}
    systemInitializer = SystemInitializer(src_folder, dst_folder, head=4, max_queue_size=4, single_batch_size=100)
    preprocessor = Preprocessor(crop_size=None)
    transformer = Processor(process_methods_with_params)
    entropyCalculator = EntropyCalculator(color_weight=None)
    dataSaver = DataSaver(dst_folder, methods=list(process_methods_with_params.keys()))

    # Initialize PipelineManager
    pipeline = PipelineManager(systemInitializer, preprocessor, transformer,
                               entropyCalculator, dataSaver)

    # Run
    pipeline.runPipeline()