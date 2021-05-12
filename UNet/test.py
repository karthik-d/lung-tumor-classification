from helpers import *
from unet import get_model

def run_test():
	metrics = list()

	model = get_model()
	model.summary()
	model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=metrics)

	weight_file = get_recent_weight_file()
	if(weight_file is None):
		print("Model not trained")
		return None
	print("Loaded", weight_file)
	model.load_weights(weight_file)

	test_img_paths = get_test_data_paths()
	test_generator = InputSequencer(
		test_img_paths
	)

	model.predict_generator(test_generator)