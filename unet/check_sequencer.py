from helpers import *
from matplotlib import pyplot as plot

def run_check_sequencer():
	img_generator = InputSequencer(get_train_data_paths())
	for i in range(10):
		img,mask = img_generator[i]

		img_axis = plot.subplot(1, 2, 1)
		img_axis.imshow(img[0,:,:])

		mask_axis = plot.subplot(1, 2, 2)
		mask_axis.imshow(mask[0,:,:,0], cmap='gray')

		plot.show()
