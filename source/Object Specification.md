# Object Specification
## Description
We want to reorgnize the code into object-oriented program style. This will benifit for maintence and speed up development.
## Program pipeline
When the program is executed, raw images are loaded from the disk and go through the preprocessor.
And then transformations are applied to images, entropy is computed.
After that, postprocess is done.
## Object
- **Image**: To represent each image and its attributes.
- **Preprocessor**: To handle all preprocessing tasks on images.
- **Transformer**: To apply different types of transformations on preprocessed images.
- **EntropyCalculator**: To compute different types of entropy.
- **Postprocessor**: To handle all post-processing tasks including data reorganization.
- **DataSaver**: To save the processed data, possibly to disk or a database.
- **PipelineManager**: To coordinate the loader, preprocessing, transformation, entropy calculation, postprocessing, and saving steps.

### 0.ImageLoader
#### Attributes:
- image_directory: Location, can be a str or a list
- image_format: a tuple, default is ('png')
#### Methods:
- load_images: To load batch images

### 1. Image
This object represents one image and its attributes.
#### Attributes:
- rawData (the opened image pixels)
- preprocessedData (image as np array)
- transformedData (np array)
- entropyResults (dictionary) {'method': entropy}
#### Methods:
- load ()

### 2. Preprocessor
This object handles all preprocessing tasks on images.
#### Attributes:

### 3. Transformer
#### Attributes:

### 4. EntropyCalculator
#### Attributes:

### 5. Postprocessor
#### Attributes:

### 6. DataSaver
#### Attributes:

### 7. PipelineManager(Console Window Running, not GUI)
coordinate the loader, preprocessing, transformation, entropy calculation, postprocessing, and saving steps.
#### Attributes:
#### Methods:
- process_single_image: To run a cycle on a image
- runPipeline: To control the program pipeline
- runParallelPipeline: To control the program pipeline, but runs in parallel