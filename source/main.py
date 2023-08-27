from Preprocessor import Preprocessor
from Processor import Processor
from EntropyCalculator import EntropyCalculator
from Postprocessor import Postprocessor
from DataSaver import DataSaver
from PipelineManager import PipelineManager
from ImageLoader import ImageLoader
from datetime import datetime

def main():
    # System Configuration
    datasets = {'china': "../datasets/satellite/china",
                'usa': "../datasets/satellite/usa",
                'argentina': "../datasets/satellite/argentina",
                'satellite': "../datasets/satellite",
                "classified": "../datasets/classified_pictures"}

    process_methods_with_params = {'dft': None, 'naive': None, 'dwt': {'wavelet':'db1', 'level':1}}
    process_methods_with_params = {'laplace': None, 'joint_red_green': None,
                                     'lbp': None, 'lbp_gabor': None, 'RGBCM': None,
                                     'dft': None, 'naive': None, 'dwt': {'wavelet':'db1', 'level':1}}

    m_name = '-'.join(process_methods_with_params.keys())
    dst_folder = f'../entropy_results/m={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    postprocessor_methods = []

    # Initialize other components
    imageLoader = ImageLoader(image_directory=datasets['classified'], head=10)
    preprocessor = Preprocessor(crop_size=None)
    transformer = Processor(process_methods_with_params)
    entropyCalculator = EntropyCalculator(color_weight=None)
    postprocessor = Postprocessor(postprocessor_methods)
    dataSaver = DataSaver(destination=dst_folder, methods=list(process_methods_with_params.keys()))

    # Initialize PipelineManager
    pipeline = PipelineManager(imageLoader, preprocessor, transformer,
                               entropyCalculator, postprocessor, dataSaver)

    # Run
    pipeline.runPipeline()
    # pipeline.runParallelPipeline(batch_size=150)


if __name__ == '__main__':
    main()
