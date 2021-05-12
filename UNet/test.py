import keras.backend as K
import tensorflow as tf
from helpers import *
from params import *
from unet import get_model
import os

def run_test():
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
	test_size = len(test_img_paths)
	test_generator = InputSequencer(
		test_img_paths
	)

	means_acc = list()
	num_steps = test_size//BATCH_SIZE
	for i in range(num_steps):
		img_batch, mask_batch = test_generator[i]
		prediction = model.predict(img_batch)
		iou_acc = list()
		for j in range(BATCH_SIZE):
			pred_mask = prediction[j]
			true_mask = mask_batch[j]
			iou_acc.append(K.eval(mean_iou(true_mask, pred_mask)))
		step_mean = sum(iou_acc)/BATCH_SIZE
		means_acc.append(step_mean)
		print("Step {curr}/{total}: Mean-IoU = {iou}".format(curr=(i+1),total=num_steps,iou=step_mean))
	final_mean = sum(means_acc)/num_steps
	print("Mean IOU: ", final_mean)
