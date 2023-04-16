import os
import datetime
import glob
import random
import sys
import cv2


import matplotlib.pyplot as plt
import skimage.io                                     #Used for imshow function
import skimage.transform                              #Used for resize function
from skimage.morphology import label                  #Used for Run-Length-Encoding RLE to create final submission

import numpy as np
import pandas as pd
import pickle

import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model, plot_model
from keras import backend as K
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split


import cv2, os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras import *
from keras.models import *
from keras.layers import *
from keras.optimizers import *

from keras.models import *
import skimage.io
from tensorflow import keras

'''
print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Skimage      :', skimage.__version__)
print('Scikit-learn :', sklearn.__version__)
print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)
'''

img_height, img_width = (768,768)
seed = 42
random.seed = seed
np.random.seed(seed=seed)

print(random.seed)

#topDir = '/home/mirunalini/data-science-bowl-2018' #defaults to '/kaggle' in kaggle kernels, different if on own system e.g. '/home/user/kaggle/dsbowl'
#os.chdir(topDir)    #changes our python working directory to the top directory of our kaggle files
#print(os.listdir(os.path.join(topDir, 'input')))  #see what's in the input folder (where data is in)

topDir='./unet_trainset/lung_aca/'
topDir1 = './unet_testset/lung_aca/'

#train_path = os.path.join(topDir, 'images/')  #path to training data file/folder
#test_path = os.path.join(topDir1, 'images/')   #path to test data file/folder

def normalize_img(image):
    return image/255



#Y_train = get_Y_data(train_path, output_shape=(img_height,img_width))



#f_data = open('img_data.pkl', 'wb')
"""
X_train = list()
for path in os.listdir(topDir+'images/'):
    X_data_new = skimage.io.imread(topDir+'images/'+path)
    #X_train.append(normalize_img(X_data_new))
    X_train.append(X_data_new)
X_train = np.array(X_train, dtype='float32')
X_train = np.sort(X_train)
#pickle.dump(X_train, f_data)
"""

"""
#X_train = X_train.sort()
#print(X_train.shape)
y_train = list()
for path in os.listdir(topDir+'masks/'):
	#print(path)
	#y_data_new = skimage.io.imread(topDir+'masks/'+path) OB
    y_data_new = cv2.imread(topDir+'masks/'+path, 0)
    y_data_new = skimage.transform.resize(y_data_new, output_shape=y_data_new.shape+(1,), mode='constant',preserve_range = True)
    y_train.append(normalize_img(y_data_new))
Y_train = np.array(y_train)
Y_train = np.sort(Y_train)
"""

#print("SHAPES",Y_train.shape, Y_train.dtype)

#X_train = get_X_data(train_path, output_shape=(img_height,img_width))
#print(X_train.shape, X_train.dtype)

def keras_model(img_width=256, img_height=256):
    '''
    Modified from https://keunwoochoi.wordpress.com/2017/10/11/u-net-on-keras-2-0/
    '''
    n_ch_exps = [4, 5, 6, 7, 8, 9]   #the n-th deep channel's exponent i.e. 2**n 16,32,64,128,256
    k_size = (3, 3)                  #size of filter kernel
    k_init = 'he_normal'             #kernel initializer

    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (3, img_width, img_height)
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
        input_shape = (img_width, img_height, 3)

    inp = Input(shape=input_shape)

    encodeds = []

    # encoderkes the path to a directory & generates batches of augmented data.
    enc = inp
    print(n_ch_exps)
    for l_idx, n_ch in enumerate(n_ch_exps):
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        enc = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(enc)
        encodeds.append(enc)
        #print(l_idx, enc)
        if n_ch < n_ch_exps[-1]:  #do not run max pooling on the last encoding/downsampling step
            enc = MaxPooling2D(pool_size=(2,2))(enc)
        #enc = Dropout(0.1*l_idx,)(enc)

    # decoder
    dec = enc
    print(n_ch_exps[::-1][1:])
    decoder_n_chs = n_ch_exps[::-1][1:]
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
        dec = Conv2DTranspose(filters=2**n_ch, kernel_size=k_size, strides=(2,2), activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
        dec = Dropout(0.1*l_idx)(dec)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)
        dec = Conv2D(filters=2**n_ch, kernel_size=k_size, activation='relu', padding='same', kernel_initializer=k_init)(dec)

    outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(dec)

    model = Model(inputs=[inp], outputs=[outp])

    return model




# Custom IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)





# Set some model compile parameters
optimizer = 'adam'
loss      = bce_dice_loss
metrics   = [mean_iou]

# Compile our model
model = keras_model(img_width=img_width, img_height=img_height)
model.summary()

# For more GPUs
#if num_gpus > 1:
#    model = multi_gpu_model(model, gpus=num_gpus)

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def get_train_test_augmented(X_data=None, Y_data=None, validation_split=0.1, batch_size=32, seed=seed):
    """
    X_train, X_test, Y_train, Y_test = train_test_split(X_data,
                                                        Y_data,
                                                        train_size=1-validation_split,
                                                        test_size=validation_split,
                                                        random_state=seed)
                                                        """

    # Image data generator distortion options
    print("CHECK")
    data_gen_args = dict(rotation_range=45.0,
                        width_shift_range=45.0,
                        height_shift_range=0.1,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect')
    data_gen_args_mask = dict(rotation_range=45.0,
                        width_shift_range=45.0,
                        height_shift_range=0.1,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect',
                        preprocessing_function=normalize_img)


    '''
    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)


    # Test data, no data augmentation, but we create a generator anyway
    X_datagen_val = ImageDataGenerator()
    Y_datagen_val = ImageDataGenerator()
    X_datagen_val.fit(X_test, augment=True, seed=seed)
    Y_datagen_val.fit(Y_test, augment=True, seed=seed)
    X_test_augmented = X_datagen_val.flow(X_test, batch_size=batch_size, shuffle=True, seed=seed)
    Y_test_augmented = Y_datagen_val.flow(Y_test, batch_size=batch_size, shuffle=True, seed=seed)
	'''

    seed = 909 # (IMPORTANT) to transform image and corresponding mask with same augmentation parameter.
    image_datagen = ImageDataGenerator(**data_gen_args) # custom fuction for each image you can use resnet one too.
    mask_datagen = ImageDataGenerator(**data_gen_args_mask)  # to make mask as feedable formate (256,256,1)

    image_generator =image_datagen.flow_from_directory(topDir+"images/",
                                                        class_mode=None, seed=seed, batch_size=batch_size,
                                                        target_size=(768, 768))
    print(image_generator.next().shape)

    mask_generator = mask_datagen.flow_from_directory(topDir+"masks/", color_mode="grayscale", batch_size=batch_size,
                                                       class_mode=None, seed=seed,
                                                       target_size=(768, 768))
    print(mask_generator.next().shape)

    train_generator = zip(image_generator, mask_generator)
    #test_generator = zip(image_generator_test, mask_generator_test)

    # combine generators into one which yields image and masks
    #train_generator = zip(X_train_augmented, Y_train_augmented)
    #test_generator = zip(X_test_augmented, Y_test_augmented)

    #return train_generator, X_train, X_test, Y_train, Y_test
    return train_generator

    #return X_train, X_test, Y_train, Y_test


batch_size = 8

#train_generator, test_generator, X_train, X_val, Y_train, Y_val = get_train_test_augmented(X_data=X_train, Y_data=Y_train, validation_split=0.1, batch_size=batch_size)
#X_train, X_val, Y_train, Y_val = get_train_test_augmented(X_data=X_train, Y_data=Y_train, validation_split=0.1, batch_size=batch_size)
#train_generator, X_train, X_val, Y_train, Y_val = get_train_test_augmented(X_data=None, Y_data=None, validation_split=0.1, batch_size=batch_size)
train_generator = get_train_test_augmented(X_data=None, Y_data=None, validation_split=0.1, batch_size=batch_size)
# increase epoch on your own machine
#model.fit_generator(train_generator, validation_data=test_generator, validation_steps=batch_size/2, steps_per_epoch=len(X_train)/(batch_size*2), epochs=1)# callbacks=[plot_losses])
model.fit_generator(train_generator, validation_steps=batch_size/2, steps_per_epoch=4000//(batch_size*2), epochs=10)# callbacks=[plot_losses])
#model.fit_generator(train_generator, validation_steps=batch_size/2, epochs=10)# callbacks=[plot_losses])


# Save the model weights to a hdf5 file
#if num_gpus > 1:
    #Refer to https://stackoverflow.com/questions/41342098/keras-load-checkpoint-weights-hdf5-generated-by-multiple-gpus
    #model.summary()
#    model_out = model.layers[-2]  #get second last layer in multi_gpu_model i.e. model.get_layer('model_1')
#else:
model_out = model
model_out.save_weights(filepath=topDir+"working/model-weights.hdf5")

# Reload the model
model_loaded = keras_model(img_width=img_width, img_height=img_height)
model_loaded.load_weights(topDir+"working/model-weights.hdf5")


# Get test data
#X_test = get_X_data(test_path, output_shape=(img_height,img_width))

X_test = list()
for path in os.listdir(topDir1+'images/'):
    #print(path)

    X_data_new = skimage.io.imread(topDir1+'images/'+path)
    X_test.append(normalize_img(X_data_new))
X_test = np.array(X_test)
X_test = np.sort(X_test)
#X_train = X_train.sort()
#print(X_train.shape)
y_test = list()
for path in os.listdir(topDir1+'masks/'):
    #print(path)
    #y_data_new = skimage.io.imread(topDir+'masks/'+path)
    y_data_new = cv2.imread(topDir1+'masks/'+path, 0)
    y_data_new = skimage.transform.resize(y_data_new, output_shape=y_data_new.shape+(1,), mode='constant', preserve_range=True)

    #print(y_data_new)
    y_test.append(normalize_img(y_data_new))
Y_test = np.array(y_test)
Y_test = np.sort(Y_test)
print("SHAPES",Y_test.shape, Y_test.dtype)

Y_hat = model_loaded.predict(X_test, verbose=1)
Y_hat.shape


id = 1
print(X_test[id].shape)
skimage.io.imshow(X_test[id])
plt.show()
skimage.io.imshow(Y_hat[id][:,:,0])
plt.imshow(Y_hat[id], cmap='gray', vmin=0, vmax=255)
plt.show()

print(Y_hat[id].shape)
plt.show()
