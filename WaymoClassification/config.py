import os
import pathlib

# Note: These paths are relative to the folder structure of my project. Adjust accordingly depending on where your data is located
ROOT = str(pathlib.Path(os.path.dirname(os.path.abspath(__file__))).parent)
WAYMO_BASE_PATH = "/data/waymo/motion/uncompressed/tf_example"

TRAIN_FOLDER = ROOT + WAYMO_BASE_PATH + "/training/"  # Path to waymo training data
VALIDATION_FOLDER = ROOT + WAYMO_BASE_PATH + "/validation/"  # Path to waymo validation data. Note: we used 2/3 of the original validation split
TEST_FOLDER = ROOT + WAYMO_BASE_PATH + "/test/"  # Path to waymo test data. Note: we used the remaining 1/3 of the validation split. Otherwise we don't have the ground truth for the future
TAGS_FOLDER = ROOT + WAYMO_BASE_PATH + "/tags/"  # Path to save and read tags from
TAG_COUNTS_FOLDER = ROOT + WAYMO_BASE_PATH + "/tag_counts/"  # Path to save and read tags from
EVALUATIONS_FOLDER = ROOT + WAYMO_BASE_PATH + "/evaluations/"  # Path to save and read tags from
PREDICTIONS_FOLDER = ROOT + WAYMO_BASE_PATH + "/predictions/"  # Path were predictions of different models will be stored
SCALERS_PATH = ROOT + WAYMO_BASE_PATH + "/scalers/"  # Scalers (min-max) used for the LSTM

MOTION_CNN_TEST_DATA = ROOT + WAYMO_BASE_PATH + "/test_cnn/"  # Path to waymo test data, prerendered for MotionCNN.
MOTION_CNN_PATH = ROOT + "/MotionCNN/test_xception71_263000_dev_131.pth"

N_PROCESSES = 8  # Number of processes to use if multiprocessing is enabled
MULTIPROCESSING = True  # Whether to use multiprocessing
BATCH_SIZE = 128  # Batch size to load waymo dataset examples
