from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K

import matplotlib
matplotlib.use('TkAgg')
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adagrad, Nadam
from keras.utils import np_utils
from sklearn.metrics import classification_report, confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

class Model:
    def build(width, height, depth, classes):
        model = Sequential()    # initialize model
        inputShape = (height, width, depth)
        channelDim = -1
        
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            channelDim = 1

         # add o the model
         # using Depth wise separable convolutio
         # stack multiple 3x3 CONV filter

        # CONV2D(32) -> RELU -> BATCH NORM -> POOL2D -> DROPOU
        model.add(SeparableConv2D(32, (3,3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.25))

       # CONV2D(64) -> RELU -> BATCH NORM -> CONV2D(64) -> RELU -> BATCH NORM -> POOL2D -> DROPOUT
        model.add(SeparableConv2D(64, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(SeparableConv2D(64, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.25))

       # CONV2D(128) -> RELU -> BATCH NORM -> CONV2D(128) -> RELU -> BATCH NORM -> CONV2D(128) -> RELU -> BATCH NORM
        # -> POOL2D -> DROPOUT
        model.add(SeparableConv2D(128,(3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(SeparableConv2D(128, (3,3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(SeparableConv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        #model.add(Dropout(0.25))

       # FC(256) -> RELU -> BATCH NORM -> DROPOUT
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))        
        # Softmax clqssifier
        model.add(Dense(classes))       
        model.add(Activation('softmax'))
        return model

if __name__ == '__main__':
    train_data_dir = 'binary_unet_thresh_trainset/'
    test_data_dir = 'binary_unet_thresh_testset/'
    train_list = list(paths.list_images(train_data_dir))
    test_list = list(paths.list_images(test_data_dir))
    number_of_train = 12000
    number_of_test = 2976
    numEpochs = 25
    batchSize = 32
    lrRate = 1e-2
    lrRateDecay = lrRate/numEpochs

    trainAug = ImageDataGenerator(rotation_range=20,
                                rescale=1/255.0,
                                zoom_range=0.05,
                                height_shift_range=0.1,                                                             
                                width_shift_range=0.1,
                                horizontal_flip=True,                                                               
                                vertical_flip=True,
                                shear_range=0.5,                                
                                fill_mode='nearest')
    testAug = ImageDataGenerator(rescale=1/255.0)

    #Initializing training and testing generator
    trainGen = trainAug.flow_from_directory(train_data_dir,
            class_mode='categorical',
            target_size=(128,128),
            shuffle=True,                                                                                                           batch_size=batchSize,
            color_mode='rgb')
    testGen = testAug.flow_from_directory(test_data_dir,
            class_mode='categorical',    
            target_size=(128,128),
            shuffle=False,
            batch_size=batchSize,
            color_mode='rgb')

    model = Model.build(width=128, height=128, depth=3, classes=2)
    model.summary()
    opt = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-05)
    # compile the model
    # compiling with binary_crossentropy loss function as 2 classes of data
    model.compile(loss='binary_crossentropy',                                  
            optimizer=opt,
            metrics=['accuracy'])
    history = model.fit_generator(trainGen,steps_per_epoch=number_of_train // batchSize,epochs=numEpochs)
    print('Training complete')
    print('Evaluating network')
    print(np.mean(history.history['acc']))
    # make prediction on test data
    predIdx = model.predict_generator(testGen,steps=(number_of_test//batchSize))
    # grab the highest prediction indices in each sample
    predIdx = np.argmax(predIdx, axis=1)
    # print the classificaiton report
    print(classification_report(testGen.classes,predIdx, target_names=testGen.class_indices.keys()))
    # compute confusion matrix
    confusionMatrix = confusion_matrix(testGen.classes, predIdx)
    total = sum(sum(confusionMatrix))
    # compute accuracy, sensitivty, specificity
    # sensitivity measures the proportion of true positives also predicted as positives
    # Similarly, specificity measures the proportion of true negatives
    accuracy = (confusionMatrix[0,0] + confusionMatrix[1,1]) / total
    sensitivity = confusionMatrix[0,0] / (confusionMatrix[0,0] + confusionMatrix[0,1])
    specificity = confusionMatrix[1,1] / (confusionMatrix[1,0] + confusionMatrix[1,1])
    print(confusionMatrix)
    print('accuracy: {:4f}'.format(accuracy))
    print('sensitivity: {:4f}'.format(sensitivity))
    print('specificity: {:4f}'.format(specificity))
