from source.Preprocessor import Preprocessor
from source.Processor import Processor
from source.EntropyCalculator import EntropyCalculator
from source.Postprocessor import Postprocessor
from source.DataSaver import DataSaver
from source.PipelineManager import PipelineManager
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
            "noising": "../datasets/noising"}
all_methods_with_params = {'laplace': None, 'joint_red_green': None, 'joint_all': None,
                           'lbp': None, 'lbp_gabor': None, 'RGBCM': None,
                           'dft': None, 'naive': None, 'dwt': {'wavelet': 'haar', 'level': 'all'}}


def main(dst_folder=None, src_folder=None, process_methods_with_params=None):
    # System Configuration
    if process_methods_with_params is None:
        process_methods_with_params = all_methods_with_params
    if src_folder is None:
        src_folder = datasets['classified']
    if dst_folder is None:
        m_name = '-'.join(process_methods_with_params.keys())
        dst_folder = f'../entropy_results/m={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    postprocessor_methods = []

    # Initialize other components
    imageLoader = ImageLoader(image_directory=src_folder, head=None)
    preprocessor = Preprocessor(crop_size=None)
    transformer = Processor(process_methods_with_params)
    entropyCalculator = EntropyCalculator(color_weight=None)
    postprocessor = Postprocessor(postprocessor_methods)
    dataSaver = DataSaver(destination=dst_folder, methods=list(process_methods_with_params.keys())
                         , auto_save=True)

    # Initialize PipelineManager
    pipeline = PipelineManager(imageLoader, preprocessor, transformer,
                               entropyCalculator, postprocessor, dataSaver)

    # Run
    pipeline.runAutoSavePipeline(max_queue_size=50)
    # pipeline.runPipeline()
    # pipeline.runParallelPipeline(batch_size=150) # Don't use this


if __name__ == '__main__':
    main()
