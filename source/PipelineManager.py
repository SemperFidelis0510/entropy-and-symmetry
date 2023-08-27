from concurrent.futures import ThreadPoolExecutor
import time
from utils import print_progress_bar
from tqdm import tqdm
import gc
class PipelineManager:
    def __init__(self, imageloader, preprocessor, transformer, entropyCalculator, postprocessor, dataSaver):
        self.imageloader = imageloader
        self.preprocessor = preprocessor
        self.transformer = transformer
        self.entropyCalculator = entropyCalculator
        self.postprocessor = postprocessor
        self.dataSaver = dataSaver

    def process_single_image(self, image):
        self.preprocessor.applyPreprocessing(image)
        self.transformer.applyTransform(image)
        self.entropyCalculator.calculateEntropy(image)
        self.postprocessor.applyPostprocessing(image)
        return image

    def sortImages(self, images):
        # Implement your sorting logic here.
        sorted_images = sorted(images, key=lambda x: x.entropyResults)
        return sorted_images

    def runPipeline(self):
        image_objects = self.imageloader.load_images()

        # Step 1: Apply preprocessing, transformations, and entropy calculations on all images
        n = len(image_objects)
        start_time = time.time()
        for index, image_object in enumerate(image_objects):
            self.preprocessor.applyPreprocessing(image_object)
            self.transformer.applyTransform(image_object)
            self.entropyCalculator.calculateEntropy(image_object)
            self.postprocessor.applyPostprocessing(image_object)
            print_progress_bar('Entropy calculation', index+1, n, start_time=start_time)
        print(f'\nEntropy calculation done.')
        # Step 2: Sort the processed images based on some specification
        sorted_images = self.sortImages(image_objects)

        # Step 3: Save the sorted images
        for index, image_object in enumerate(sorted_images):
            self.dataSaver.save(index, image_object)

    def runParallelPipeline(self):
        images = self.imageloader.load_images()
        num_images = len(images)
        # Step 1: Parallel processing of all images
        with ThreadPoolExecutor() as executor:
            processed_images = list(tqdm(executor.map(self.process_single_image, images), total=num_images))

        # Step 2: Sort the processed images based on some specification
        sorted_images = self.sortImages(processed_images)

        # Step 3: Save the sorted images
        for index, image_object in enumerate(sorted_images):
            self.dataSaver.save(index, image_object)


