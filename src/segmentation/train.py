from unet import get_model
from helpers import *
from params import *

# "sparse" version of categorical_crossentropy is used
# because target data is integers.
def run_train():
	metrics = list()

	model = get_model()
	model.summary()
	model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=metrics)


	callbacks = [
		keras.callbacks.ModelCheckpoint(os.path.join(WEIGHTS_PATH, "weights.{epoch:02d}.hdf5"), 
										save_best_only=False,
										save_weights_only=True)
	]

	train_img_paths = get_train_data_paths()
	train_generator = InputSequencer(
		train_img_paths
	)
	model.fit_generator(train_generator, epochs=NUM_EPOCHS, callbacks=callbacks)