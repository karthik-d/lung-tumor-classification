from keras.preprocessing.image import load_img
from keras import layers
import tensorflow as tf
import keras
import numpy as np

import os

from params import *

IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
TRAIN_PATH_IMAGES = os.path.join(TRAIN_PATH, 'images', 'data')
TRAIN_PATH_MASKS = os.path.join(TRAIN_PATH, 'masks', 'data')

def prepare_train_data():
	img_paths = [ os.path.join(TRAIN_PATH_IMAGES, img_name)
        			for img_name in sorted(os.listdir(TRAIN_PATH_IMAGES)) ]
	mask_paths = [ os.path.join(TRAIN_PATH_MASKS, mask_name)
                for mask_name in sorted(os.listdir(TRAIN_PATH_IMAGES)) ]  
	return img_paths, mask_paths

"""
## `Sequence` class to load & vectorize batches of data
"""

class InputSequencer(keras.utils.Sequence):

	def __init__(self, image_paths, mask_paths):
		self.BATCH_SIZE = BATCH_SIZE
		self.IMG_SIZE = IMG_SIZE
		self.image_paths = image_paths
		self.mask_paths = mask_paths

	def __len__(self):
		return len(self.image_paths) // self.BATCH_SIZE

	def __getitem__(self, idx):
		"""Returns tuple (input, target) correspond to batch #idx."""
		i = idx * self.BATCH_SIZE
		batch_image_paths = self.image_paths[i : i + self.BATCH_SIZE]
		batch_mask_paths = self.mask_paths[i : i + self.BATCH_SIZE]

		x = np.zeros((self.BATCH_SIZE,) + self.IMG_SIZE + (3,), dtype="float32")
		for j, path in enumerate(batch_image_paths):
			img = load_img(path, target_size=self.IMG_SIZE)
			x[j] = img
		y = np.zeros((self.BATCH_SIZE,) + self.IMG_SIZE + (1,), dtype="uint8")
		for j, path in enumerate(batch_mask_paths):
			img = load_img(path, target_size=self.IMG_SIZE, color_mode="grayscale")
			y[j] = np.expand_dims(img, 2)
			# Label pixels with 255 as 1, and 0 as 0
			y[j] //= 255
		return x, y


"""
## U-Net Xception-style model
"""


def get_model():
    inputs = keras.Input(shape=IMG_SIZE + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(NUM_CLASSES, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.models.Model(inputs, outputs)
    return model

def mean_iou(y_true, y_pred):
	prec = []
	#for t in np.arange(0.5, 1.0, 0.05):
	for t in np.arange(0.5):
		y_pred_ = tf.to_int32(y_pred > t)
		score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
		keras.backend.get_session().run(tf.local_variables_initializer())
		with tf.control_dependencies([up_opt]):
			score = tf.identity(score)
		prec.append(score)
	return keras.backend.mean(keras.backend.stack(prec), axis=0)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    intersection = keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras.backend.sum(y_true_f) + keras.backend.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

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

"""
## Train the model
"""

metrics = [
    mean_iou
]

model = get_model()
model.summary()
# "sparse" version of categorical_crossentropy
# because target data is integers.
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=metrics)

callbacks = [
    keras.callbacks.ModelCheckpoint("alt_weights.h5", save_best_only=True)
]

image_paths, mask_paths = prepare_train_data()

train_generator = InputSequencer(
	image_paths, mask_paths
)

model.fit(train_generator, epochs=NUM_EPOCHS, callbacks=callbacks)

'''
def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    display(img)
'''