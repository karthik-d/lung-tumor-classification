from unet import get_model
from  helpers import *
from params import *

"""
# VALIDATION SPLIT
import random

# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = OxfordPets(
    BATCH_SIZE, IMG_SIZE, train_input_img_paths, train_target_img_paths
)
val_gen = OxfordPets(BATCH_SIZE, IMG_SIZE, val_input_img_paths, val_target_img_paths)
"""


"""metrics = [
    mean_iou
]"""


# "sparse" version of categorical_crossentropy is used
# because target data is integers.
def run_train():
	metrics = list()

	model = get_model()
	model.summary()
	model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=metrics)


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