import tensorflow as tf
from matplotlib import pyplot as plot
import numpy as np
import os
from helpers import *
from params import *
from unet import get_model

def run_predict(n=10):
	model = get_model()
	model.summary()
	model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

	weight_file = get_recent_weight_file()
	if(weight_file is None):
		print("Model not trained")
		return None
	print("Loaded", weight_file)
	model.load_weights(os.path.join(WEIGHTS_PATH, weight_file))

	for class_name in CLASS_NAMES:
		predict_img_paths = get_predict_data_paths(class_name=class_name)
		predict_generator = PredictSequencer(
			predict_img_paths,
			shuffle=False
		)

		"""
		# Random 'n' images
		test_images = list()
		for i in range(n):
			img, mask = test_generator[i]
			test_images.append(img[0])

		predictions = model.predict(np.array(test_images))

		for idx, mask in enumerate(predictions):
			plot.imshow(test_images[idx])
			plot_img_mask(test_images[idx], mask)
		"""

		counter = 0
		for batch_ip in predict_generator:
			img_ip_batch, img_path_batch = batch_ip
			img_shape = img_ip_batch[0].shape
			mask_op_batch = model.predict(img_ip_batch)	
			for idx, img_path in enumerate(img_path_batch):
				img_name = img_path.split('/')[-1]
				# Store the mask
				write_path_mask = os.path.join(
					PREDICT_PATH_MASKS, 
					img_name
					).format(
						class_name = class_name
					)
				mask = get_mask_img_from_prediction(mask_op_batch[idx])
				cv2.imwrite(write_path_mask, mask)
				# Store the image extracts
				write_path_extract = os.path.join(
					PREDICT_PATH_EXTRACTS, 
					img_name
					).format(
						class_name = class_name
					)
				img_ip = img_ip_batch[idx]
				img_extract = np.zeros(img_shape)
				img_extract[mask==1] = img_ip[mask==1]
				cv2.imwrite(write_path_extract, img_extract)
			counter += 1
			if(counter%10==0):
				print(class_name, counter)