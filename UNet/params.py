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
#CLASS_NAMES = [ "benign", "malignent"]

WEIGHTS_PATH = os.path.join(BASE_PATH, "dynamic")

TRAIN_PATH = os.path.join(DATA_PATH, 'train', '{class_name}')
TRAIN_PATH_IMAGES = os.path.join(TRAIN_PATH, 'images', 'data')
TRAIN_PATH_MASKS = os.path.join(TRAIN_PATH, 'masks', 'data')

TEST_PATH = os.path.join(DATA_PATH, 'test', '{class_name}')
TEST_PATH_IMAGES = os.path.join(TEST_PATH, 'images', 'data')
TEST_PATH_MASKS = os.path.join(TEST_PATH, 'masks', 'data')

# ~/histopathology/nuclie_segmentation/NucleusMask/NucleusMask/UNet/Dataset/unet_trainset/lung_aca/images/data
# ~/histopathology/second_review/trainset/lung_aca
PREDICT_PATH = os.path.join(
	'/'
	'home', 
	'mirunap', 
	'histopathology', 
	'nuclie_segmentation', 
	'NucleusMask',
	'NucleusMask',
	'UNet',
	'Dataset',
	'unet_trainset',
	'{class_name}',
	'{r_type}',
	'data'
)

PREDICT_PATH_IMAGES = PREDICT_PATH.format(r_type='images', class_name='{class_name}')
#PREDICT_PATH_MASKS = PREDICT_PATH.format(r_type='extracts')
PREDICT_PATH_EXTRACTS = PREDICT_PATH.format(r_type='extracts', class_name='{class_name}')

# Create Result directories if they don't exist
for class_name in CLASS_NAMES:
	extract_path = PREDICT_PATH_EXTRACTS.format(class_name=class_name)
	if not os.path.isdir(extract_path):
		os.makedirs(extract_path)

# For the Image Data Generator
GENERATOR_SEED = 100

# Train Parameters
BATCH_SIZE = 16
#NUM_INPUTS = len(os.listdir(os.path.join(TRAIN_PATH, 'images', 'data')))
NUM_EPOCHS = 20
NUM_CLASSES = 2

