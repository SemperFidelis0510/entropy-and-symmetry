from Preprocessor import Preprocessor
from Transformer import Transformer
from EntropyCalculator import EntropyCalculator
from Postprocessor import Postprocessor
from DataSaver import DataSaver
from PipelineManager import PipelineManager
from ImageLoader import ImageLoader
def main():
    # System Configuration
    preprocess_methods = []
    transform_methods = ['dft', 'naive']
    calent_methods = []
    postprocessor_methods = []
    image_directory = ''

    # Initialize other components
    imageLoader = ImageLoader(image_directory)
    preprocessor = Preprocessor(preprocess_methods)
    transformer = Transformer(transform_methods)
    entropyCalculator = EntropyCalculator(calent_methods)
    postprocessor = Postprocessor(postprocessor_methods)
    dataSaver = DataSaver(destination='some_path', format='png')

    # Initialize PipelineManager
    pipeline = PipelineManager(imageLoader, preprocessor, transformer,
                               entropyCalculator, postprocessor, dataSaver)

    # Run
    pipeline.runPipeline()

    # Run parallel
    # pipeline.runParallelPipeline()