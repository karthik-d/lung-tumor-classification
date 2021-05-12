import keras
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plot
import os
import cv2

from params import *

class InputSequencer(keras.utils.Sequence):

	def __init__(self, image_paths):
		self.BATCH_SIZE = BATCH_SIZE
		self.IMG_SIZE = IMG_SIZE
		self.image_paths = image_paths

	def __len__(self):
		return len(self.image_paths) // self.BATCH_SIZE

	def __getitem__(self, idx):
		"""Returns tuple (input, target) correspond to batch #idx."""
		i = idx * self.BATCH_SIZE
		batch_image_paths = self.image_paths[i : i + self.BATCH_SIZE]

		x = np.zeros((self.BATCH_SIZE,) + self.IMG_SIZE + (3,), dtype="float32")
		y = np.zeros((self.BATCH_SIZE,) + self.IMG_SIZE + (1,), dtype="uint8")
		for j, path in enumerate(batch_image_paths):
			# Load Images (x-data)
			img = cv2.imread(path)
			img = img/255  # Normalize
			img = cv2.resize(img, self.IMG_SIZE)
			x[j] = img
			# Load Masks (y-data)
			mask_path = os.path.join(TRAIN_PATH_MASKS, get_mask_name(os.path.split(path)[1]))
			mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			mask = cv2.resize(mask, self.IMG_SIZE)
			mask[mask<127] = 1
			mask[mask>=127] = 0   # Split into classes
			y[j] = np.expand_dims(mask, 2)
		return x, y

def get_train_data_paths():
	img_paths = [ os.path.join(TRAIN_PATH_IMAGES, img_name)
        			for img_name in sorted(os.listdir(TRAIN_PATH_IMAGES)) ]
	return img_paths

def get_test_data_paths():
	img_paths = [ os.path.join(TEST_PATH_IMAGES, img_name)
        			for img_name in sorted(os.listdir(TEST_PATH_IMAGES)) ]
	return img_paths

def get_mask_name(img_name):
	img_name, ext = img_name.split('.')
	mask_name = img_name + '_white'
	return mask_name+ext 

def get_mask_from_prediction(prediction):
	mask = np.argmax(prediction, axis=-1)
	mask = np.expand_dims(mask, axis=-1)
	return mask

def get_mask_img_from_prediction(prediction):
	mask = np.argmax(prediction, axis=-1)
	return mask

def mean_iou(y_true, y_pred):
    y_pred = get_mask_from_prediction(y_pred)
    y_true = K.cast(K.equal(y_true, 1), K.floatx())
    y_pred = K.cast(K.equal(y_pred, 1), K.floatx())
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)

def display_mask(prediction):
	mask = np.argmax(prediction, axis=-1)
	mask = np.expand_dims(mask, axis=-1)
	plot.imshow(mask, cmap='gray')
	plot.show()

def get_recent_weight_file():
	weight_file = None
	max_epoch = -1
	for filename in os.listdir(WEIGHTS_PATH):
		parts = filename.split('.')
		if('hdf5' in parts):
			epoch = int(parts[1])
			if(epoch>max_epoch):
				weight_file = filename
				max_epoch = epoch
	return weight_file

def plot_img_mask(img, mask):
	img_axis = plot.subplot(1, 2, 1)
	img_axis.imshow(img)
	mask_axis = plot.subplot(1, 2, 2)
	mask = get_mask_img_from_prediction(mask)
	mask_axis.imshow(mask, cmap='gray')
	plot.show()