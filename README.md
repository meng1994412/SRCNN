# SRCNN
## Objectives
* Implemented super resolution convolutional neural networks (SRCNN) and applied super resolution to input images.

## Packages Used
* Python 3.6
* [OpenCV](https://docs.opencv.org/3.4.4/) 4.0.0
* [keras](https://keras.io/) 2.1.0
* [Tensorflow](https://www.tensorflow.org/install/) 1.13.0
* [cuda toolkit](https://developer.nvidia.com/cuda-toolkit) 10.0
* [cuDNN](https://developer.nvidia.com/cudnn) 7.4.2
* [Imutils](https://github.com/jrosebr1/imutils)
* [NumPy](http://www.numpy.org/)
* [SciPy](https://www.scipy.org/scipylib/index.html)
* [Matplotlib](https://matplotlib.org/)

## Approaches
The dataset is 100 images from [UKBench](https://archive.org/details/ukbench) dataset.

The `sr_config.py` ([check here](https://github.com/meng1994412/SRCNN/blob/master/config/sr_config.py)) under `config/` directory stores the configurations for the project, including path to the input images, path to the temporary output directories, path to the `HDF5` files, and etc.

The `srcnn,py` ([check here](https://github.com/meng1994412/SRCNN/blob/master/pipeline/nn/conv/srcnn.py)) under `pipeline/nn/conv/` directory construct the super resolution convolutional neural network, which is fully convolutional. In this convolutional neural network, we train for filters, not accuracy. We are concerned with actual filters learn by SRCNN which will enable us to upscale an image.

The goal is to make SRCNN learn how to reconstruct high resolution patchesfrom low resolution input ones. Thus, we are going to construct two sets of image patches, including a low resolution patch that will be the input to the network, and a high resolution patch that will be target for the network to predict.

The `build_dataset.py` ([check here](https://github.com/meng1994412/SRCNN/blob/master/build_dataset.py)) builds a dataset of low and high resolution input patches.

The `train.py` ([check here](https://github.com/meng1994412/SRCNN/blob/master/train.py)) trains a network to learn to map the low resolution patches to their high resolution counterparts.

The `resize.py` ([check here](https://github.com/meng1994412/SRCNN/blob/master/resize.py)) utilizes loops over the input patches of a low resolution images, passes them through the network, and then creates the output high resolution image from the predicted patches.

There are two helper classes for building dataset or training process:
The `hdf5datasetwriter.py` ([check here](https://github.com/meng1994412/SRCNN/blob/master/pipeline/io/hdf5datasetwriter.py)) under `pipeline/io/` directory, defines a class that help to write raw images or features into `HDF5` dataset.

The `hdf5datasetgenerator.py` ([check here](https://github.com/meng1994412/SRCNN/blob/master/pipeline/io/hdf5datasetgenerator.py)) under `pipeline/io/` directory yields batches of images and labels from `HDF5` dataset. This class can help to facilitate our ability to work with datasets that are too big to fit into memory.

## Results
Figure 1 shows the original image. And Figure 2 increases the size of original image about two times but without applying any SRCNN, as a baseline image. Figure 3 demonstrates the image after applying SRCNN, which is about same size as Figure 2.

<img src="https://github.com/meng1994412/SRCNN/blob/master/results/beagle.jpg" width="300">

Figure 1: Original image.

<img src="https://github.com/meng1994412/SRCNN/blob/master/results/beagle_baseline.png" width="600">

Figure 2: Baseline image (2x bigger than original image), before applying SRCNN.

<img src="https://github.com/meng1994412/SRCNN/blob/master/results/beagle_output.png" width="600">

Figure 3: Output image, after applying SRCNN.
