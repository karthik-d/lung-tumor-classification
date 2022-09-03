import os

# Dimensions for Input Image (to the model)
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Train and Test paths
BASE_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), 
    *((os.path.pardir,)*2)
)
DATA_PATH = os.path.join(
    BASE_PATH, 
    'data',
    'dataset',
)
SEG_DATA_PATH = os.path.join(
    DATA_PATH,
    'segmentation'
)
CLASS_NAMES = [ "lung_aca", "lung_n", "lung_scc" ]

WEIGHTS_PATH = os.path.join(BASE_PATH, "dynamic")

TRAIN_PATH = os.path.join(SEG_DATA_PATH, 'trainset', '{class_name}')
TRAIN_PATH_IMAGES = os.path.join(TRAIN_PATH, 'images', 'data')
TRAIN_PATH_MASKS = os.path.join(TRAIN_PATH, 'masks', 'data')

TEST_PATH = os.path.join(SEG_DATA_PATH, 'testset', '{class_name}')
TEST_PATH_IMAGES = os.path.join(TEST_PATH, 'images', 'data')
TEST_PATH_MASKS = os.path.join(TEST_PATH, 'masks', 'data')

SAMPLE_PATH = os.path.join(SEG_DATA_PATH, 'samples', '{class_name}')
SAMPLE_PATH_IMAGES = os.path.join(SAMPLE_PATH, 'images', 'data')
SAMPLE_PATH_MASKS = os.path.join(SAMPLE_PATH, 'masks', 'data')

'''
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
	'unet_testset',
	'{class_name}',
	'{r_type}',
	'data'
)'''

PREDICT_PATH = os.path.join(
	DATA_PATH,
    'trainset'
	'{class_name}'
)

PREDICT_PATH_IMAGES = os.path.join(PREDICT_PATH, 'wsi')
PREDICT_PATH_EXTRACTS = os.path.join(PREDICT_PATH, 'nuclei')

# Create Result directories if they don't exist
for class_name in CLASS_NAMES:
	extract_path = PREDICT_PATH_EXTRACTS.format(class_name=class_name)
	if not os.path.isdir(extract_path):
		os.makedirs(extract_path)

# For the Image Data Generator
GENERATOR_SEED = 100

# Train Parameters
BATCH_SIZE = 1
#NUM_INPUTS = len(os.listdir(os.path.join(TRAIN_PATH, 'images', 'data')))
NUM_EPOCHS = 20
NUM_CLASSES = 2

