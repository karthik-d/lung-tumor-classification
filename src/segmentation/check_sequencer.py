from helpers import *
from matplotlib import pyplot as plot

def run_check_sequencer():
	# img_generator = InputSequencer(get_train_data_paths())
	img_generator = InputSequencer(get_sample_data_paths())

	for i in range(12):
		img, mask, paths = img_generator[i]
		img = img.astype(np.uint8)
		mask_3ch = np.array([np.stack((mask.squeeze(), mask.squeeze(), mask.squeeze()), axis=-1)])
		# print(mask)

		img[mask_3ch==0] = 0

		img_axis = plot.subplot(1, 2, 1)
		img_axis.imshow(img.squeeze())

		mask_axis = plot.subplot(1, 2, 2)
		mask_axis.imshow(mask[0,:,:,0], cmap='gray')

		print(paths)

		plot.show()
