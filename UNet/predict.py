from helpers import *
from matplotlib import pyplot as plot
import numpy as np

def run_predict(n=10):
	model.load_weights(get_recent_weight_file())

	test_img_paths = get_test_data_paths()
	test_generator = InputSequencer(
		test_img_paths
	)

	# Random 'n' images
	test_images = list()
	for i in range(n):
		img, mask = train_generator[i]
		test_images.append(img[0])
		mask *= 255
		plot.imshow(mask[0,:,:,0], cmap='gray')
		plot.show()
		plot.imshow(img[0,:,:,:])
		plot.show()		

	predictions = model.predict(np.array(test_images))

	for idx, mask in enumerate(predictions):
		plot.imshow(test_images[idx])
		plot.imshow()
		display_mask(mask)