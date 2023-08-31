## Objective

The main objective of this summer project is to investigate the relationship between image textures, entropy measures,
and image classification. The project will begin by exploring images with repeating patterns and various symmetries to
understand how entropy relates to structured textures. Subsequently, we will apply our knowledge to analyze real-world
satellite images of Earth, investigating dependencies on location, time, and zoom levels. Additionally, we will extend
our scope to investigate video data, studying the temporal evolution of entropy in moving images. Furthermore, the
project will include an analysis of subimages within a larger image to examine patterns of entropy at different scales.
This project will provide hands-on experience in Python programming, image processing, and the calculation and
application of various entropy measures.

## Stages

### Stage 1: Data Collection - Repeating Patterns

The initial stage will focus on collecting a diverse dataset of images containing repeating patterns with different
symmetries. We will use sources like public image repositories or image generation services to obtain images with
controlled textures. Emphasis will be placed on selecting images with various symmetries, such as rotational,
translational, or reflectional symmetries. Additionally, we will generate images of fractals using known algorithms with
controlled parameters. These fractal images will be valuable for later investigations into entropy.

### Stage 2: Developing Entropy Calculation Algorithms and Image Preprocessing

In this stage, our focus will be twofold: developing algorithms to calculate various entropy measures for the images
with symmetries and creating any necessary image preprocessing algorithms. We will explore and implement methods to
compute entropy of the greyscale representation, entropy of the color histogram, and entropy based on different image
transformations. The goal is to have a set of functional and efficient algorithms that can calculate entropy values for
the images with symmetries accurately.
Additionally, we will design and implement image preprocessing algorithms to ensure that the images are in a suitable
format and quality for entropy analysis. Image preprocessing may involve tasks such as resizing, normalization, noise
reduction, or contrast enhancement.
By the end of this stage, we will have a toolbox of entropy calculation algorithms, along with any necessary image
preprocessing algorithms, all ready to be utilized in the subsequent stages of the project.

### Stage 3: Texture Analysis and Classification - Images with Symmetries

In Stage 3, we will utilize the entropy calculation algorithms developed in the previous stage to analyze the images
with symmetries. Using the calculated entropy values, we will order and classify the images based on their complexity
and structure. By assigning semantic meaning to different entropy measures, we aim to understand their significance in
describing image complexities.
This texture analysis and classification will provide valuable insights into the potential applications of entropy
measures in image analysis tasks. We will evaluate the classification accuracy and examine how different entropy
measures contribute to distinguishing various symmetries in the images, paving the way for practical use of entropy in
image classification scenarios.

### Stage 4: Subimage Analysis - Examining Entropy Patterns

In this crucial stage, we will investigate the entropy of subimages within a larger image. We will divide the images
into smaller, overlapping, or non-overlapping subimages and compute the entropy for each subimage using different
entropy measures. By analyzing patterns of entropy at different scales, we will gain a deeper understanding of how image
structure and complexity vary within the larger context. This analysis will provide valuable insights into the role of
entropy in capturing localized image features and will be essential for the subsequent analysis.

### Stage 5: Investigating Videos - Temporal Analysis of Entropy

In this stage, we will expand the project to video data and investigate the temporal evolution of entropy in moving
images. We will analyze a collection of video clips and compute various entropy measures over time, exploring how the
information content and complexity of images change dynamically. This stage will provide insights into the role of
entropy in video analysis and its potential applications.

### Stage 6: Analysis of Real-World Satellite Images

Having gained proficiency in texture analysis and classification, we will proceed to the final stage of the project. We
will apply our knowledge and algorithms to analyze real-world satellite images of Earth. The focus will be on studying
the textures and patterns present in the satellite images and calculating their entropy using different entropy
measures. Additionally, we will investigate the dependencies of entropy on the geographical location, time of
acquisition, and zoom levels of the images. This comprehensive analysis will offer valuable insights into the
information content of satellite imagery and its spatial and spectral characteristics.

## Throughout the Project

We will be encouraged to document our progress, insights, and results. Regular meetings will be scheduled to provide
guidance, support, and foster collaboration among the students. By investigating repeating patterns, real-world
satellite images, temporal video analysis, and subimage entropy patterns using various entropy measures, this project
will offer a comprehensive exploration of entropy's role in texture analysis and its application in practical image
classification and video analysis scenarios.

## Using the Repository

To ensure smooth collaboration and organization within the project, we have established some guidelines on how to use
this repository effectively.

### Resources and Images

For adding resources, datasets, or images related to the project, you can directly push the files to the `main` branch.
This allows easy access to shared resources for all project members.

### Scripts and Code Development

When working on scripts or any code development for the project, we encourage the following approaches:

1. **Working on the `main` Branch**: For small changes or quick additions, you can create or modify files directly on
   the `main` branch. This is suitable for minor adjustments or small code snippets.

2. **Branching Out**: If you are working on a more substantial script or feature, it is advisable to create a new branch
   from the `main` branch. Name the branch with a descriptive title related to the work you will be doing. This allows
   you to work on your task independently without affecting the `main` branch.

### Personal Walkthroughs

For individual walk-throughs or experiments that you wish to perform separately, each student should create their own
branch. This helps in isolating individual work and avoids potential conflicts when working in parallel.

When you are satisfied with the changes and additions made in your branch, you can create a pull request to merge it
back into the `main` branch. This way, we can review the changes as a team before incorporating them into the main
project.

By adhering to these guidelines, we can ensure a well-organized and collaborative environment for our project. It also
helps in tracking and managing the progress of various tasks efficiently.

