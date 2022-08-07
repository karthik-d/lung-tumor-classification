from histomicstk.preprocessing.color_normalization.deconvolution_based_normalization import (
	deconvolution_based_normalization
)	
import keras
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plot
import os
import cv2

from params import *

class InputSequencer(keras.utils.Sequence):

	def __init__(self, image_paths, shuffle=True):
		self.BATCH_SIZE = BATCH_SIZE
		self.IMG_SIZE = IMG_SIZE
		self.shuffle = shuffle
		self.image_paths = image_paths
		self.indexes = np.arange(len(self.image_paths))
		self.on_epoch_end()

	def on_epoch_end(self):
		if(self.shuffle):
			np.random.shuffle(self.indexes)

	def __len__(self):
		return len(self.image_paths) // self.BATCH_SIZE

	def __getitem__(self, idx):
		"""Returns tuple (input, target) correspond to batch #idx."""
		i = idx * self.BATCH_SIZE
		indices = self.indexes[i : i + self.BATCH_SIZE]
		
		batch_image_paths = [ self.image_paths[idx] 
								for idx in indices ]

		x = np.zeros((self.BATCH_SIZE,) + self.IMG_SIZE + (3,), dtype="float32")
		y = np.zeros((self.BATCH_SIZE,) + self.IMG_SIZE + (1,), dtype="uint8")
		for j, path in enumerate(batch_image_paths):
			# Load Images (x-data)
			img = cv2.imread(path)
			img = cv2.resize(img, self.IMG_SIZE)

			# Stain unnixing + color normalization
			stain_unmixing_routine_params = {
				'stains': ['hematoxylin', 'eosin'],
				'stain_unmixing_method': 'macenko_pca',
			}
			img_normalized = deconvolution_based_normalization(
				img,
				stain_unmixing_routine_params=stain_unmixing_routine_params
			)

			"""
			# CLAHE preprocess
			clahe_applicator = cv2.createCLAHE(clipLimit=3)
			temp_img = None
			img = clahe_applicator.apply(img)
			img = cv2.normalize(img, temp_img, 0, 255, cv2.NORM_MINMAX)
			img = img/255  # Normalize
			"""

			x[j] = img_normalized
			# Load Masks (y-data)
			mask_path = os.path.join(
				TRAIN_PATH_MASKS.format(class_name=get_classname_from_path(path)),
				get_maskname_from_path(path)
			)
			mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			mask = cv2.resize(mask, self.IMG_SIZE)
			mask[mask<127] = 1
			mask[mask>=127] = 0   # Split into classes
			y[j] = np.expand_dims(mask, 2)

		return x, y


class PredictSequencer(keras.utils.Sequence):

	def __init__(self, image_paths, shuffle=False):
		self.BATCH_SIZE = BATCH_SIZE
		self.IMG_SIZE = IMG_SIZE
		self.shuffle = shuffle
		self.image_paths = image_paths
		self.indexes = np.arange(len(self.image_paths))
		self.on_epoch_end()

	def on_epoch_end(self):
		if(self.shuffle):
			np.random.shuffle(self.indexes)

	def __len__(self):
		return len(self.image_paths) // self.BATCH_SIZE

	def __getitem__(self, idx):
		"""
		Returns (input_imgs, input_paths) corresponding to batch #idx
		"""
		
		i = idx * self.BATCH_SIZE
		indices = self.indexes[i : i + self.BATCH_SIZE]
		
		batch_image_paths = [ self.image_paths[idx] 
								for idx in indices ]

		x = np.zeros((self.BATCH_SIZE,) + self.IMG_SIZE + (3,), dtype="float32")
		for j, path in enumerate(batch_image_paths):
			# Load Images (x-data)
			img = cv2.imread(path)
			img = img/255  # Normalize
			img = cv2.resize(img, self.IMG_SIZE)
			x[j] = img

		return x, batch_image_paths


def get_classname_from_path(path):
	for class_name in CLASS_NAMES:
		if class_name in path:
			return class_name 
	return None

def get_train_data_paths():
	img_paths = list()
	for class_name in CLASS_NAMES:
		img_dir_path = TRAIN_PATH_IMAGES.format(class_name=class_name)
		print(img_dir_path)
		img_paths.extend([ os.path.join(img_dir_path, img_name)
							for img_name in os.listdir(img_dir_path) ])
	return img_paths

def get_test_data_paths():
	img_paths = list()
	for class_name in CLASS_NAMES:
		img_dir_path = TEST_PATH_IMAGES.format(class_name=class_name)
		img_paths.extend([ os.path.join(img_dir_path, img_name)
							for img_name in os.listdir(img_dir_path) ])
	return img_paths

def get_predict_data_paths(class_name):
	# SET PREDICT_PATH first, in "params.py"
	img_dir_path = PREDICT_PATH_IMAGES.format(class_name=class_name)
	img_paths = [
		os.path.join(img_dir_path, img_name)
		for img_name in os.listdir(img_dir_path) 
	]
	return img_paths

def get_maskname_from_path(path):
	img = os.path.split(path)[1]
	img_name, ext = img.split('.')
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