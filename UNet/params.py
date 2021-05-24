import os

# Dimensions for Input Image (to the model)
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Train and Test paths
BASE_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(BASE_PATH, 'Dataset')
CLASS_NAMES = [ "lung_aca", "lung_n", "lung_scc" ]
#CLASS_NAMES = [ "benign", "malignaent"]

WEIGHTS_PATH = os.path.join(BASE_PATH, "dynamic")

TRAIN_PATH = os.path.join(DATA_PATH, 'train', '{class_name}')
TRAIN_PATH_IMAGES = os.path.join(TRAIN_PATH, 'images', 'data')
TRAIN_PATH_MASKS = os.path.join(TRAIN_PATH, 'masks', 'data')

TEST_PATH = os.path.join(DATA_PATH, 'test', '{class_name}')
TEST_PATH_IMAGES = os.path.join(TEST_PATH, 'images', 'data')
TEST_PATH_MASKS = os.path.join(TEST_PATH, 'masks', 'data')

# ~/histopathology/second_review/trainset/lung_aca
PREDICT_PATH = os.path.join(
	'/'
	'home', 
	'mirunap', 
	'histopathology', 
	'first_review', 
	'trainset'
)
PREDICT_PATH_IMAGES = os.path.join(PREDICT_PATH, '{class_name}')
PREDICT_PATH_MASKS = os.path.join(PREDICT_PATH, '{class_name}_unet_mask')
PREDICT_PATH_EXTRACTS = os.path.join(PREDICT_PATH, '{class_name}_unet_extracts')

# Create Result directories if they don't exist
for class_name in CLASS_NAMES:
	mask_path = PREDICT_PATH_MASKS.format(class_name=class_name)
	if not os.path.isdir(mask_path):
		os.mkdir(mask_path)
	extract_path = PREDICT_PATH_EXTRACTS.format(class_name=class_name)
	if not os.path.isdir(extract_path):
		os.mkdir(extract_path)

# For the Image Data Generator
GENERATOR_SEED = 100

# Train Parameters
BATCH_SIZE = 16
#NUM_INPUTS = len(os.listdir(os.path.join(TRAIN_PATH, 'images', 'data')))
NUM_EPOCHS = 20
NUM_CLASSES = 2