from source.Preprocessor import Preprocessor
from source.Processor import Processor
from source.EntropyCalculator import EntropyCalculator
from source.DataSaver import DataSaver
from source.PipelineManager import PipelineManager
from source.SystemInitializer import SystemInitializer
from source.ImageLoader import ImageLoader
import warnings

warnings.filterwarnings("ignore")

datasets = {'china': "../datasets/satellite/china",
            'usa': "../datasets/satellite/usa",
            'argentina': "../datasets/satellite/dftargentina",
            'satellite': "../datasets/satellite",
            "classified": "../datasets/classified_pictures",
            "fix_noise": "../datasets/fixed_noise.bmp",
            "test_satellite": "../datasets/classified_pictures/Agriculture/map_image_38.7904_ -99.2209_20230828191958.png",
            "noising": "../datasets/noising"}
all_methods_with_params = {'laplace': None, 'joint_red_green': None, 'joint_all': None,
                           'lbp': None, 'lbp_gabor': None, 'RGBCM': None, 'hist':None,
                           'dft': None, 'naive': None, 'dwt': {'wavelet': 'haar', 'level': 'all'}}
channels = {
    "rgb": "rgb",
    "hsb": "hsb",
    "YCbCr": "YCbCr"
}
def reset_ent_norm():
    print('reset entropy norm')
    src_folder = datasets["fix_noise"]
    dst_folder = "data"
    imageLoader = ImageLoader()
    systemInitializer = SystemInitializer(src_folder, dst_folder)
    preprocessor = Preprocessor()
    processor = Processor(all_methods_with_params, level=3)
    entropyCalculator = EntropyCalculator(color_weight=None, reset_norm=True)
    dataSaver = DataSaver(dst_folder, methods=list(all_methods_with_params.keys()))

    # Initialize PipelineManager
    pipeline = PipelineManager(systemInitializer, imageLoader, preprocessor,
                               processor,entropyCalculator, dataSaver)
    pipeline.runPipeline()
def main(dst_folder=None, src_folder=None, process_methods_with_params=None,
         head=None, max_queue_size=None, single_batch_size=None, callback=None, processed_level=None):
    # System Configuration
    if process_methods_with_params is None:
        process_methods_with_params = all_methods_with_params
    if src_folder is None:
        src_folder = datasets['fix_noise']
    if dst_folder is None:
        dst_folder = f'../processed/testonly'
    if head is None:
        head = None
    if max_queue_size is None:
        max_queue_size = 4
    if single_batch_size is None:
        single_batch_size = 4
    if processed_level is None:
        processed_level = 2
    systemInitializer = SystemInitializer(src_folder, dst_folder, head=head, max_queue_size=max_queue_size,
                                        single_batch_size=single_batch_size)
    imageLoader = ImageLoader(callback=callback)
    preprocessor = Preprocessor(crop_size=None)
    processor = Processor(process_methods_with_params, level=processed_level)
    entropyCalculator = EntropyCalculator(color_weight=None)
    dataSaver = DataSaver(dst_folder, methods=list(process_methods_with_params.keys()))

    # Initialize PipelineManager
    pipeline = PipelineManager(systemInitializer, imageLoader, preprocessor,
                               processor, entropyCalculator, dataSaver, callback=callback)
    pipeline.runPipeline()

def test_and_analyze():
    for methods, param in all_methods_with_params.items():
        method_para = {methods:param}
        dst_folder = f"../processed/test_and_analyze/{methods}"
        src_folder = datasets["noising"]
        main(dst_folder, src_folder, method_para, single_batch_size=25)

def main_gui(dst_folder=None, src_folder=None, process_methods_with_params=None,
         head=None, max_queue_size=None, single_batch_size=None, callback=None, processed_level=None):
    # System Configuration
    if process_methods_with_params is None:
        process_methods_with_params = all_methods_with_params
    if src_folder is None:
        src_folder = datasets['classified']
    if dst_folder is None:
        dst_folder = f'../processed/testonly'
    if head is None:
        head = None
    if max_queue_size is None:
        max_queue_size = 4
    if single_batch_size is None:
        single_batch_size = 4
    if processed_level is None:
        processed_level = 2
    systemInitializer = SystemInitializer(src_folder, dst_folder, head=head, max_queue_size=max_queue_size,
                                        single_batch_size=single_batch_size)
    imageLoader = ImageLoader(callback=callback)
    preprocessor = Preprocessor(crop_size=None)
    processor = Processor(process_methods_with_params, level=processed_level)
    entropyCalculator = EntropyCalculator(color_weight=None)
    dataSaver = DataSaver(dst_folder, methods=list(process_methods_with_params.keys()))

    # Initialize PipelineManager
    pipeline = PipelineManager(systemInitializer, imageLoader, preprocessor,
                               processor, entropyCalculator, dataSaver, callback=callback)
    pipeline.runPipeline()

if __name__ == '__main__':
    # main()
    # reset_ent_norm()
    # test_and_analyze()
    src_folder = datasets['fix_noise']
    dst_folder = '../processed/tests'
    max_queue_size = 5      # For every how many results will be saved to disk
    single_batch_size = 100 # How many image in process for each batch
    processed_level = 3      # level, 0, 1, 2,3
    main(dst_folder=dst_folder, src_folder=src_folder, max_queue_size=max_queue_size,
         single_batch_size=single_batch_size, processed_level=processed_level, process_methods_with_params=all_methods_with_params)
