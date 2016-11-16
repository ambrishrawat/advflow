#!/usr/bin/env python

''' 
Source
https://github.com/scienceai/neocortex/blob/master/examples/cifar10_cnn/cifar10_cnn_example.ipynb
'''

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU, ParametricSoftplus
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np


batch_size = 64
nb_classes = 10
nb_epoch = 100

img_channels = 3
img_rows = 32
img_cols = 32

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=None,
    width_shift_range=None,
    height_shift_range=None,
    horizontal_flip=True,
    vertical_flip=False)

'''
batch = 0
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=2048):
    print(batch, end='...', flush=True)
    X_train = np.vstack((X_train, X_batch))
    y_train = np.vstack((y_train, y_batch))
    batch += 1
'''

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255



model = Sequential()

model.add(Convolution2D(32, 3, 3,border_mode = 'same',
                        input_shape=( img_rows, img_cols, img_channels)))
model.add(LeakyReLU(alpha=0.2))
model.add(Convolution2D(32, 3, 3))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3,3,border_mode = 'same' ))
model.add(LeakyReLU(alpha=0.2))
model.add(Convolution2D(64, 3, 3))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(128, 3, 3, border_mode = 'same'))
model.add(LeakyReLU(alpha=0.2))
model.add(Convolution2D(128, 3, 3))
model.add(LeakyReLU(alpha=0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode='categorical', metrics = ['accuracy'])
    
checkpointer = ModelCheckpoint(filepath='cifar10_cnn_keras_weights.hdf5', verbose=1, save_best_only=True)
earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

model.fit_generator(datagen.flow(X_train, Y_train, 
          batch_size=batch_size), 
          samples_per_epoch=2*X_train.shape[0],
          nb_epoch=nb_epoch, 
          show_accuracy=True,
          validation_data=(X_test, Y_test),
          callbacks=[checkpointer, earlystopping])

from keras.models import model_from_json
model_json = model.to_json()
with open("models/model_gpu3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/model_gpu3.h5")
print("Saved model to disk")












