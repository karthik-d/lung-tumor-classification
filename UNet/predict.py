import tensorflow as tf
from matplotlib import pyplot as plot
import numpy as np
import os
from helpers import *
from params import *
from unet import get_model

def run_predict(n=10):
	metrics = [
		mean_iou
	]

	model = get_model()
	model.summary()
	model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=metrics)

	weight_file = get_recent_weight_file()
	if(weight_file is None):
		print("Model not trained")
		return None
	print("Loaded", weight_file)
	model.load_weights(os.path.join(WEIGHTS_PATH, weight_file))

	test_img_paths = get_test_data_paths()
	test_generator = InputSequencer(
		test_img_paths
	)

	# Random 'n' images
	test_images = list()
	for i in range(n):
		img, mask = test_generator[i]
		test_images.append(img[0])

	predictions = model.predict(np.array(test_images))

	for idx, mask in enumerate(predictions):
		plot.imshow(test_images[idx])
		plot_img_mask(test_images[idx], mask)