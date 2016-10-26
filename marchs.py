#Netowrk-in-network model

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from keras import initializations
from keras.regularizers import l2, activity_l2

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave
from keras.preprocessing.image import ImageDataGenerator



def nin(input_shape):

    model = Sequential()

    #init_001 = 
    model.add(Convolution2D(96, 11, 11, activation='relu', name='conv1_1', 
                            init='normal',subsample=(4,4), input_shape = input_shape))
    model.add(Convolution2D(96, 1, 1, activation='relu', name='conv1_1', 
                            init='normal',subsample=(1,1), input_shape = input_shape))
    model.add(Convolution2D(96, 1, 1, activation='relu', name='conv1_1',   
                            init='normal',subsample=(1,1), input_shape = input_shape))    
    model.add(MaxPooling2D(pool_size=(3, 2)))




    model.add(Convolution2D(96, 11, 11, activation='relu', name='conv1_1', 
                            init='normal',subsample=(4,4), input_shape = input_shape))
    model.add(Convolution2D(96, 1, 1, activation='relu', name='conv1_1', 
                            init='normal',subsample=(1,1), input_shape = input_shape))
    model.add(Convolution2D(96, 1, 1, activation='relu', name='conv1_1',   
                            init='normal',subsample=(1,1), input_shape = input_shape))    
    model.add(MaxPooling2D(pool_size=(3, 2)))


    model.add(Dropout(0.25))


def vgg19():
    model = Sequential()

    model.add(ZeroPadding2D((1,1),input_shape=(64,64,3)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_4'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_4'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_4'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(200, name='dense_3'))
    model.add(Activation("sigmoid"))

    return model

