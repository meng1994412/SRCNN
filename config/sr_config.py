# import packages
import os

# define the path to the input images that will be used to build the training sets
INPUT_IMAGES = "../datasets/ukbench/ukbench100"

# define the path to the temporary output directories
BASE_OUTPUT = "../datasets/ukbench/output"
IMAGES = os.path.sep.join([BASE_OUTPUT, "images"])
LABELS = os.path.sep.join([BASE_OUTPUT, "labels"])

# define the path to the HDF5 files
INPUTS_DB = os.path.sep.join([BASE_OUTPUT, "inputs.hdf5"])
OUTPUTS_DB = os.path.sep.join([BASE_OUTPUT, "outputs.hdf5"])

# define the path to the output model file and plot file
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "srcnn.model"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])

# initialize the batch size and number of epochs for training
BATCH_SIZE = 128
NUM_EPOCHS = 10

# initialize the scale along with the input width and height dimensions to SRCNN
SCALE = 2.0
INPUT_DIM = 33

# the label size should be the output spatial dimensions of the SRCNN
# while paading ensures we properly crop the label ROI
LABEL_SIZE = 21
PAD = int((INPUT_DIM - LABEL_SIZE) / 2.0)

# the stride controls the step size of our sliding window
STRIDE = 14
