import keras
import numpy as np
from matplotlib import pyplot as plot

import os
import cv2

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

def mean_iou(y_true, y_pred):
	prec = []
	y_pred = get_mask_from_prediction(y_pred)
	#for t in np.arange(0.5, 1.0, 0.05):
	for t in np.arange(0.5):
		y_pred_ = tf.to_int32(y_pred > t)
		score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
		keras.backend.get_session().run(tf.local_variables_initializer())
		with tf.control_dependencies([up_opt]):
			score = tf.identity(score)
		prec.append(score)
	return keras.backend.mean(keras.backend.stack(prec), axis=0)

def display_mask(prediction):
	mask = np.argmax(prediction, axis=-1)
	mask = np.expand_dims(mask, axis=-1)
	plot.imshow(mask, cmap='gray')
	plot.show()

def get_recent_weight_file():
	