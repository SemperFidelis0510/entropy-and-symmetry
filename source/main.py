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
            "test_satellite": "../datasets/classified_pictures/city/map_image_21.9073_ 42.0248_20230827210356.png",
            "noising": "../datasets/noising"}
all_methods_with_params = {'laplace': None, 'joint_red_green': None, 'joint_all': None,
                           'lbp': None, 'lbp_gabor': None, 'RGBCM': None,
                           'dft': None, 'naive': None, 'dwt': {'wavelet': 'haar', 'level': 'all'}}

def reset_ent_norm():
    print('reset entropy norm')
    src_folder = datasets["fix_noise"]
    dst_folder = "data"
    imageLoader = ImageLoader()
    systemInitializer = SystemInitializer(src_folder, dst_folder, head=None)
    preprocessor = Preprocessor(crop_size=None)
    transformer = Processor(all_methods_with_params)
    entropyCalculator = EntropyCalculator(color_weight=None, reset_norm=True)
    dataSaver = DataSaver(dst_folder, methods=list(all_methods_with_params.keys()))

    # Initialize PipelineManager
    pipeline = PipelineManager(systemInitializer, imageLoader, preprocessor,
                               transformer,entropyCalculator, dataSaver)
    pipeline.runPipeline()
    print('please check the result and rename to entropy_norm.json')
def main(dst_folder=None, src_folder=None, process_methods_with_params=None,
         head=None, max_queue_size=None, single_batch_size=None, callback=None):
    # System Configuration
    if process_methods_with_params is None:
        process_methods_with_params = {'dft': None} #all_methods_with_params
    if src_folder is None:
        src_folder = datasets['fix_noise']
    if dst_folder is None:
        m_name = '-'.join(process_methods_with_params.keys())
        dst_folder = f'../processed/localtests'
    if head is None:
        head = 4
    if max_queue_size is None:
        max_queue_size = 40
    if single_batch_size is None:
        single_batch_size = 1000
    processed_level = 1
    systemInitializer = SystemInitializer(src_folder, dst_folder, head=head,
                                          max_queue_size=max_queue_size, single_batch_size=single_batch_size)
    imageLoader = ImageLoader(callback=callback)
    preprocessor = Preprocessor(crop_size=None)
    processor = Processor(process_methods_with_params, level=processed_level)
    entropyCalculator = EntropyCalculator(color_weight=None)
    dataSaver = DataSaver(dst_folder, methods=list(process_methods_with_params.keys()))

    # Initialize PipelineManager
    pipeline = PipelineManager(systemInitializer, imageLoader, preprocessor,
                               processor, entropyCalculator, dataSaver, callback=callback)
    pipeline.runPipeline()


def main_gui(dst_folder=None, src_folder=None, process_methods_with_params=None,
         head=None, max_queue_size=None, single_batch_size=None, callback=None):
    # System Configuration
    if process_methods_with_params is None:
        process_methods_with_params = all_methods_with_params
    if src_folder is None:
        src_folder = datasets['classified']
    if dst_folder is None:
        m_name = '-'.join(process_methods_with_params.keys())
        dst_folder = f'../processed/results'
    if head is None:
        head = None
    if max_queue_size is None:
        max_queue_size = 40
    if single_batch_size is None:
        single_batch_size = 1000

    systemInitializer = SystemInitializer(src_folder, dst_folder, head=head,
                                          max_queue_size=max_queue_size, single_batch_size=single_batch_size)
    imageLoader = ImageLoader(callback=callback)
    preprocessor = Preprocessor(crop_size=None)
    transformer = Processor(process_methods_with_params)
    entropyCalculator = EntropyCalculator(color_weight=None)
    dataSaver = DataSaver(dst_folder, methods=list(process_methods_with_params.keys()))

    # Initialize PipelineManager
    pipeline = PipelineManager(systemInitializer, imageLoader, preprocessor,
                               transformer,entropyCalculator, dataSaver)
    pipeline.runPipeline()

if __name__ == '__main__':
    main()
