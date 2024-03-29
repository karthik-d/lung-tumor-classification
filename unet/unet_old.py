import os
import sys
import random
import warnings

import cv2
from matplotlib import pyplot as plot
import numpy as np
import pandas as pd

from params import *

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

def normalize_image(img):
    return img / 255

def coupled_traindata_generator(generators):
    for(img, mask) in zip(generators[0], generators[1]):
        yield img[0], mask[0]

def make_train_generators():
    print(os.path.join(TRAIN_PATH, 'images'))
    image_data_generator = ImageDataGenerator(preprocessing_function=normalize_image
                                                ).flow_from_directory(
                                                    os.path.join(TRAIN_PATH, 'images'), 
                                                    batch_size = BATCH_SIZE, 
                                                    target_size = (IMG_WIDTH, IMG_HEIGHT), 
                                                    seed = GENERATOR_SEED,
                                                    color_mode="rgb")

    print(os.path.join(TRAIN_PATH, 'masks'))
    mask_data_generator = ImageDataGenerator(preprocessing_function=normalize_image
                                                ).flow_from_directory(
                                                    os.path.join(TRAIN_PATH, 'masks'), 
                                                    batch_size = BATCH_SIZE, 
                                                    target_size = (IMG_WIDTH, IMG_HEIGHT), 
                                                    seed = GENERATOR_SEED,
                                                    color_mode="grayscale")
    return image_data_generator, mask_data_generator

"""
input_generator = coupled_traindata_generator(make_train_generators())
for data_pair in input_generator:
    print(data_pair[1])  # Prints ONE MASK
    break
exit()
"""

def make_model():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    #c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    #c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    #c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    #c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    #c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    #c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    #c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    #c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
   
def mean_iou(y_true, y_pred):
    prec = []
    #for t in np.arange(0.5, 1.0, 0.05):
	for t in np.arange(0.5):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

# Loss function
loss = bce_dice_loss

# Metrics
metrics = [
    mean_iou
]

model = make_model()
model.compile(optimizer='adam', loss=loss, metrics=metrics)
model.summary()


input_generator = coupled_traindata_generator(make_train_generators())
model.fit_generator(input_generator,
                    epochs=NUM_EPOCHS,
                    steps_per_epoch=(NUM_INPUTS // BATCH_SIZE))

model.save_weights("model_weights")

# Prediction

model.load_weights("model_weights")

test_images = list()
for img_name in os.listdir(os.path.join(TRAIN_PATH, 'images', 'data')):
    img = cv2.imread(os.path.join(TRAIN_PATH, 'images', 'data', img_name))
    img = normalize_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    test_images.append(img)
    break

predictions = model.predict(np.array(test_images))

for mask in predictions:
	mask[mask<0.5] = 0
	mask[mask>=0.5] = 255
	plot.imshow(mask[:,:,0], cmap='gray')
	plot.show()

