import os.path

from source.Preprocessor import Preprocessor
from source.Processor import Processor
from source.EntropyCalculator import EntropyCalculator
from source.DataSaver import DataSaver
from source.PipelineManager import PipelineManager
from source.SystemInitializer import SystemInitializer
from source.ImageLoader import ImageLoader
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

datasets = {'china': "../datasets/satellite/china",
            'usa': "../datasets/satellite/usa",
            'argentina': "../datasets/satellite/argentina",
            'satellite': "../datasets/satellite",
            "classified": "../datasets/classified_pictures",
            "fix_noise": "../datasets/fixed_noise.bmp",
            "test_satellite": "../datasets/classified_pictures/detailed nature/map_image_40.1618_ -102.3710_20230821155009.png",
            "noising": "../datasets/noising"}
all_methods_with_params = {'laplace': None, 'joint_red_green': None, 'joint_all': None,
                           'lbp': None, 'lbp_gabor': None, 'RGBCM': None,
                           'dft': None, 'naive': None, 'dwt': {'wavelet': 'haar', 'level': 'all'}}
channels = {
    "rgb": "rgb",
    "hsb": "hsb",
    "YCbCr": "YCbCr"
}
def reset_ent_norm(preprocess_channels):
    print('reset entropy norm')
    src_folder = datasets["fix_noise"]
    dst_folder = f"data/{preprocess_channels}"
    imageLoader = ImageLoader()
    systemInitializer = SystemInitializer(src_folder, dst_folder, preprocess_channels=preprocess_channels)
    preprocessor = Preprocessor(channels=preprocess_channels)
    processor = Processor(all_methods_with_params, level=2)
    entropyCalculator = EntropyCalculator(color_weight=None, reset_norm=True)
    dataSaver = DataSaver(dst_folder, methods=list(all_methods_with_params.keys()))

    # Initialize PipelineManager
    pipeline = PipelineManager(systemInitializer, imageLoader, preprocessor,
                               processor,entropyCalculator, dataSaver)
    pipeline.runPipeline()
    print('please check the result and rename to entropy_norm.json')
def main(dst_folder=None, src_folder=None, process_methods_with_params=None,
         head=None, max_queue_size=None, single_batch_size=None, callback=None,
         preprocess_channels=None, processed_level=None):
    # System Configuration
    if process_methods_with_params is None:
        process_methods_with_params = all_methods_with_params
    if src_folder is None:
        src_folder = datasets['test_satellite']
    if dst_folder is None:
        dst_folder = f'../processed/localtests/sub'
    if head is None:
        head = None
    if max_queue_size is None:
        max_queue_size = 4
    if single_batch_size is None:
        single_batch_size = 4
    if preprocess_channels is None:
       preprocess_channels = channels['YCbCr']
    if processed_level is None:
        processed_level = 2
    ent_norm_path = f"data/{preprocess_channels}/entropy_results.json"
    systemInitializer = SystemInitializer(src_folder, dst_folder, head=head,max_queue_size=max_queue_size,
                                        single_batch_size=single_batch_size, preprocess_channels=preprocess_channels
                                          , ent_norm_path=ent_norm_path)
    imageLoader = ImageLoader(callback=callback)
    preprocessor = Preprocessor(crop_size=None, channels=preprocess_channels)
    processor = Processor(process_methods_with_params, level=processed_level)
    entropyCalculator = EntropyCalculator(color_weight=None, ent_norm_path=ent_norm_path)
    dataSaver = DataSaver(dst_folder, methods=list(process_methods_with_params.keys()))

    # Initialize PipelineManager
    pipeline = PipelineManager(systemInitializer, imageLoader, preprocessor,
                               processor, entropyCalculator, dataSaver, callback=callback)
    pipeline.runPipeline()

def main_gui(dst_folder=None, src_folder=None, process_methods_with_params=None,
         head=None, max_queue_size=None, single_batch_size=None, callback=None,
             preprocess_channels=None, processed_level=None):
    # System Configuration
    if process_methods_with_params is None:
        process_methods_with_params = all_methods_with_params
    if src_folder is None:
        src_folder = datasets['classified']
    if dst_folder is None:
        m_name = '-'.join(process_methods_with_params.keys())
        dst_folder = f'../processed/gui_results'
    if head is None:
        head = None
    if max_queue_size is None:
        max_queue_size = 30
    if single_batch_size is None:
        single_batch_size = 1000
    if preprocess_channels is None:
        preprocess_channels = channels['rgb']
    if processed_level is None:
        processed_level = 2
    ent_norm_path = f"data/{preprocess_channels}/entropy_results.json"
    systemInitializer = SystemInitializer(src_folder, dst_folder, head=head, max_queue_size=max_queue_size,
                                          single_batch_size=single_batch_size, preprocess_channels=preprocess_channels
                                          , ent_norm_path=ent_norm_path)
    imageLoader = ImageLoader(callback=callback)
    preprocessor = Preprocessor(crop_size=None, channels=preprocess_channels)
    processor = Processor(process_methods_with_params, level=processed_level)
    entropyCalculator = EntropyCalculator(color_weight=None, ent_norm_path=ent_norm_path)
    dataSaver = DataSaver(dst_folder, methods=list(process_methods_with_params.keys()))

    # Initialize PipelineManager
    pipeline = PipelineManager(systemInitializer, imageLoader, preprocessor,
                               processor, entropyCalculator, dataSaver, callback=callback)
    pipeline.runPipeline()

if __name__ == '__main__':
    main()
    # preprocess_channels = channels["YCbCr"]
    # reset_ent_norm(preprocess_channels)
