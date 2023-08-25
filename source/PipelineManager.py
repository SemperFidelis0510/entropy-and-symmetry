from concurrent.futures import ThreadPoolExecutor
import threading
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
        images = self.imageloader.load_images()

        # Step 1: Apply preprocessing, transformations, and entropy calculations on all images
        for image in images:
            self.preprocessor.applyPreprocessing(image)
            self.transformer.applyTransform(image)
            self.entropyCalculator.calculateEntropy(image)
            self.postprocessor.applyPostprocessing(image)

        # Step 2: Sort the processed images based on some specification
        sorted_images = self.sortImages(images)

        # Step 3: Save the sorted images
        for image in sorted_images:
            self.dataSaver.save(image)

    def runParallelPipeline(self, images):
        images = self.imageloader.load_images()

        # Step 1: Parallel processing of all images
        with ThreadPoolExecutor() as executor:
            processed_images = list(executor.map(self.process_single_image, images))

        # Step 2: Sort the processed images based on some specification
        sorted_images = self.sortImages(processed_images)

        # Step 3: Save the sorted images
        for image in sorted_images:
            self.dataSaver.save(image)


