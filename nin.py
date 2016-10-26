#Netowrk-in-network model

from future__ import print_function
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



