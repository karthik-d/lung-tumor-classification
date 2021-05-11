import os

# Dimensions for Input Image (to the model)
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

# Train and Test paths
BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), "Dataset"))
CLASS_NAME = "lung_aca"
TRAIN_PATH = os.path.join(BASE_PATH, 'train', CLASS_NAME)
TEST_PATH = os.path.join(BASE_PATH, 'test', CLASS_NAME)

# For the Image Data Generator
GENERATOR_SEED = 100

# Train Parameters
BATCH_SIZE = 16
NUM_INPUTS = len(os.listdir(os.path.join(TRAIN_PATH, 'images', 'data')))
NUM_EPOCHS = 20
NUM_CLASSES = 2