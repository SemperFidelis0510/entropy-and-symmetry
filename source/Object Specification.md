# Object Specification

## Description

We want to reorgnize the code into object-oriented program style. This will benifit for maintence and speed up
development.

## Program pipeline
When the program is executed, first we will analyze system state, then batch images are loaded from the disk and go through the preprocessor.
And then different processed methods are applied to images, entropy is computed. When an image is finished, it will be added to the save queue.
After the queue is full, the results will be saved. Along the program, log will be produced to track the running procedure.
## Object
- **SystemInitializer**: To analyze the system state.
- **ImageLoader**: To load image from a given path
- **Image**: To represent each image and its attributes.
- **Preprocessor**: To handle all preprocessing tasks on images.
- **Transformer**: To apply different types of transformations on preprocessed images.
- **EntropyCalculator**: To compute different types of entropy.
- **Postprocessor**: To handle all post-processing tasks including data reorganization.
- **DataSaver**: To save the processed data, possibly to disk or a database.
- **PipelineManager**: To coordinate the loader, preprocessing, transformation, entropy calculation, postprocessing, and
  saving steps.

### 0.ImageLoader

#### Attributes:

- image_directory: Location, can be a str points to a single file or a list
- image_format: a tuple, default is ('png', 'bmp')
- image_paths: contains all image that will be loaded to memory
- head: how many images want to be loaded and process
- registry: default is 'registry.json',

#### Methods:

- load_images: To load all images from image_paths, return list of image object
- load_batch_images: To batch load images from disk and return list of image object
- prepare_path: analyze image_directory and initialize the image_paths

### 1. Image

This object represents one image and its attributes.

#### Attributes:

- rawData (the opened image pixels)
- preprocessedData (image as np array, produced by preprocessor and deleted after processed)
- processedData (np array, produced by processor and deleted after entropyCalculation)
- path (Image path, where the image loaded from disk)
- size (image size on disk)
- pixel_size (a tuple (width, height))
- entropyResults (dictionary) {'method1': entropy, 'method2': entropy}

### 2. Preprocessor

This object handles all preprocessing tasks on images.

#### Attributes:

- crop_size: To crop image into a same size

#### Methods:

- applyPreprocessing: get an image object, do preprocessing and then save relevant data to image object for further
  process

### 3. Processor

#### Attributes:

- processing_methods_with_params: {'method1': {'para1': value1, ...}, ...}

#### Methods:

- applyProcessing: process the data with different methods and then save relevant data to image object for further usage
- all the other methods: ...........

### 4. EntropyCalculator

#### Attributes:

- color_weight: (1, 1, 1), default is gray color weight

#### Methods:

- calculateEntropy: compute entropy for all methods
- entropy: helper function to compute entropy

### 5. Postprocessor

This class reserved for future usage

### 6. DataSaver

#### Attributes:

- destination
- methods
- json_path
- json_file

#### Methods:

- save: save a single image into disk
- save_ent_result: saved list of images' metadata to the json file
- save_single_ent_result: append a single image's metadata to the json file
- get_ent_result: construct json metadata of processed images

### 7. PipelineManager(Console Window Running, not GUI)

coordinate the loader, preprocessing, transformation, entropy calculation, postprocessing, and saving steps.

#### Attributes:

- imageloader
- preprocessor
- processor
- entropyCalculator
- postprocessor
- dataSaver

#### Methods:

- process_single_image: To run a cycle on a image
- runPipeline: To control the program pipeline
- runParallelPipeline: To control the program pipeline, but runs in parallel
- sortImages: sort the image by their entropyResults
- process_and_save_single_image:
- runParallelPipeline
