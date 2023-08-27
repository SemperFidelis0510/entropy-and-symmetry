from Preprocessor import Preprocessor
from Transformer import Transformer
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
                'satellite': "../datasets/satellite"}

    transform_methods_with_params = {'dft': None, 'naive': None, 'dwt': {'wavelet':'db1', 'level':1}}
    transform_methods_with_params = {'laplace': None, 'joint_red_green': None,
                                     'lbp': None, 'lbp_gabor': None, 'RGBCM': None,
                                     'dft': None, 'naive': None, 'dwt': {'wavelet':'db1', 'level':1}}

    m_name = '-'.join(transform_methods_with_params.keys())
    dst_folder = f'../entropy_results/m={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    postprocessor_methods = []

    # Initialize other components
    imageLoader = ImageLoader(image_directory=datasets['satellite'], head=10)
    preprocessor = Preprocessor(crop_size=None)
    transformer = Transformer(transform_methods_with_params)
    entropyCalculator = EntropyCalculator(color_weight=(1,1,1))
    postprocessor = Postprocessor(postprocessor_methods)
    dataSaver = DataSaver(destination=dst_folder, methods=list(transform_methods_with_params.keys()))

    # Initialize PipelineManager
    pipeline = PipelineManager(imageLoader, preprocessor, transformer,
                               entropyCalculator, postprocessor, dataSaver)

    # Run
    pipeline.runPipeline()
    # pipeline.runParallelPipeline(batch_size=150)


if __name__ == '__main__':
    main()
