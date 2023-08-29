import gc
from concurrent.futures import ThreadPoolExecutor
import time
from source.utils import print_progress_bar, open_folder
from tqdm import tqdm


class PipelineManager:
    def __init__(self, imageloader, preprocessor, processor, entropyCalculator, postprocessor, dataSaver):
        self.imageloader = imageloader
        self.preprocessor = preprocessor
        self.processor = processor
        self.entropyCalculator = entropyCalculator
        self.postprocessor = postprocessor
        self.dataSaver = dataSaver

    def process_single_image(self, image):
        self.preprocessor.applyPreprocessing(image)
        self.processor.applyProcessing(image)
        del image.preprocessedData
        self.entropyCalculator.calculateEntropy(image)
        del image.processedData
        self.postprocessor.applyPostprocessing(image)
        return image

    def sortImages(self, images):
        # Implement your sorting logic here.
        sorted_images = sorted(images, key=lambda x: x.entropyResults)
        return sorted_images

    def runPipeline(self, batch_size=100):
        image_objects = self.imageloader.load_images()

        # Step 1: Apply preprocessing, transformations, and entropy calculations on all images
        n = len(image_objects)
        start_time = time.time()
        for index, image_object in enumerate(image_objects):
            self.process_single_image(image_object)
            print_progress_bar('Entropy calculation', index + 1, n, start_time=start_time)
        print(f'\nEntropy calculation done.')
        # Step 2: Sort the processed images based on some specification
        sorted_images = self.sortImages(image_objects)

        # Step 3: Save the sorted images
        for index, image_object in enumerate(sorted_images):
            self.dataSaver.save(index, image_object)

        self.dataSaver.save_ent_result(sorted_images)

        open_folder(self.dataSaver.destination)

    def runAutoSavePipeline(self, max_queue_size=30):  # Saved result when the queue is full
        image_objects = self.imageloader.load_images()

        # Step 1: Apply preprocessing, transformations, and entropy calculations on all images
        n = len(image_objects)
        start_time = time.time()
        queue_images = []
        for index, image_object in enumerate(image_objects):
            self.process_single_image(image_object)
            queue_images.append(image_object)
            if len(queue_images) == max_queue_size:
                base_index = index - max_queue_size + 1
                for queue_index, image_object in enumerate(queue_images):
                    self.dataSaver.save(base_index + queue_index, image_object)
                self.dataSaver.auto_save_ent_result(base_index, queue_images)
                queue_images = []
            print_progress_bar('Entropy calculation', index + 1, n, start_time=start_time)
        size = len(queue_images)
        if size:  # last queue
            start_time = time.time()
            base_index = n - max_queue_size + 1
            for queue_index, image_object in enumerate(queue_images):
                self.dataSaver.save(base_index + queue_index, image_object)
                print_progress_bar('Saving last queue', queue_index + 1, size, start_time=start_time)
            self.dataSaver.auto_save_ent_result(base_index, queue_images)
        self.dataSaver.prettify_json_file()
        print(f'\nEntropy calculation done.')
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

    def runParallelPipeline(self, batch_size=300):
        images = self.imageloader.load_images()
        num_images = len(images)

        # Process and save images in batches
        with ThreadPoolExecutor() as executor:
            for i in range(0, num_images, batch_size):
                batch = images[i:i + batch_size]
                batch_indices = range(i, i + len(batch))
                list(tqdm(executor.map(self.process_and_save_single_image, batch, batch_indices), total=len(batch)))
        open_folder(self.dataSaver.destination)
