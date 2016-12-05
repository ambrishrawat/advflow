import tensorflow as tf
import os
import pandas as pd
import numpy as np

from keras.callbacks import Callback
import time
import csv
from scipy.misc import imresize, imread, imshow
import skimage
from skimage import io
import keras


from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, generic_utils


def load_img(img_filename):
    """
    Load image from the filename. Default is to load in color if
    possible.
    Args:
        img_name (string): string of the image name, relative to
            the image directory.
    Returns:
        np array of float32: an image as a numpy array of float32
    """
    #img = skimage.img_as_float(imread(
    #    img_filename,mode='RGB')).astype(np.float32)
    img = imread(img_filename,mode='RGB')
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    #'''Subtract channel vise mean'''
    #ch_mean = np.load('preprocessing/ch_mean.npy')
    #img = img - ch_mean
    return img/255



class CSVGenerator():
    '''
    Class for iterating over a pandas data frame with filenames and addresses

    TODO
    add comments
    '''

    def __init__(self,csv_location = None,
                 batch_size=32,
                 shuffle=False,
                 target_size=None,
                 nbsamples=None):
        
        if csv_location is not None:
            if nbsamples is not None:
                self.df = pd.read_csv(csv_location, nrows=nbsamples)
            else:
                self.df = pd.read_csv(csv_location)

        self.N = self.df.shape[0]
        print("number of samples loaded",self.N)

        self.batch_size = batch_size
        self.total_batches_seen = 0
        self.index_gen = self._idx_gen(self.N,batch_size,shuffle)
        self.target_size = target_size
        print(self.target_size)

    def get_data_size(self):
        return self.N

    def _idx_gen(self,N,batch_size=32,shuffle=False):
        batch_index = 0
        while 1:
            if batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)
            current_index = (batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                batch_index += 1
            else:
                current_batch_size = N - current_index
                batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def batch_gen(self):
        while 1:
            index_array, current_index, current_batch_size = self.index_gen.__next__()
            
            if self.target_size is not None:    
                img = [imresize(load_img(self.df.iloc[i]['filename']),size=self.target_size) for i in index_array]
            else:
                img = [load_img(self.df.iloc[i]['filename']) for i in index_array]
            img = np.asarray(img)
            lab = [self.df.iloc[i].iloc[1:].values.astype('float32') for i in index_array]
            lab = np.asarray(lab)
            if keras.backend.image_dim_ordering() == 'th':
                nimg, ch, h, w = img.shape[0], img.shape[3], img.shape[1], img.shape[2] 
                img = np.rollaxis(img, 2, 1).reshape(nimg, ch, h, w)

            #Convert to float as most pretrained models are trained on this
            #img = img.astype('float32')
            #img /= 255
            yield img,lab



def load_cifar_as_numpy():
    '''
    Load the cifar set as numpy arrays
    '''
    train_datagen = CSVGenerator(csv_location='./../adversarial/preprocessing/train_cifar10.csv',
                                     batch_size=10,shuffle=True,
                                     nbsamples=None)

    train_generator = train_datagen.batch_gen()
    X_train_,Y_train_ = train_generator.__next__()

    val_datagen = CSVGenerator(csv_location='./../adversarial/preprocessing/test_cifar10.csv',
                                     batch_size=10,
                                     nbsamples=None)

    val_generator = val_datagen.batch_gen()
    X_test_,Y_test_ = val_generator.__next__()



    return (X_train_,Y_train_),(X_test_,Y_test_)



class Cifar_npy_gen():

    def __init__(self,batch_size=64):
        '''
        Using keras generators
        '''


        nb_classes = 10
        
        # the data, shuffled and split between tran and test sets
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        
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

        self.Y_train = np_utils.to_categorical(y_train, nb_classes)
        self.Y_test = np_utils.to_categorical(y_test, nb_classes)


        self.X_train = X_train.astype('float32')
        self.X_test = X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255
    
        self.train_gen = datagen.flow(self.X_train, self.Y_train,
                          batch_size=batch_size)

        datagen2 = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=None,
            width_shift_range=None,
            height_shift_range=None,
            horizontal_flip=False,
            vertical_flip=False)

        self.test_gen = datagen2.flow(self.X_test, self.Y_test,
                          batch_size=batch_size,
                          shuffle=False)


def return_gen(X,Y,batch_size):
    
    datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=None,
            width_shift_range=None,
            height_shift_range=None,
            horizontal_flip=False,
            vertical_flip=False)

    return datagen.flow(X, Y,
                      batch_size=batch_size,
                      shuffle=False)


def save_npy(np_array = None,
        specs = None,
        ):
    '''
    Save numpy array given the specifications
    '''

    ''' Make the mid directory '''
    directory = os.path.join(specs['work_dir'],specs['save_id'])
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save(os.path.join(directory,specs['file_id']),np_array) 



class NPYGenerator():
    '''
    Class for iterating over a NPY array

    TODO
    add comments
    '''

    def __init__(self,img_npy=None,
            label_npy=None,
            batch_size=32,
            shuffle=False,
            nbsamples=None):
       
        self.img_npy = img_npy
        self.label_npy = label_npy
        self.N = self.img_npy.shape[0]
        if nbsamples is not None:
            self.N = nbsamples
        self.batch_size = batch_size
        self.total_batches_seen = 0
        self.index_gen = self._idx_gen(self.N,batch_size,shuffle)

    def get_data_size(self):
        return self.N

    def _idx_gen(self,N,batch_size=32,shuffle=False):
        batch_index = 0
        while 1:
            if batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)
            current_index = (batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                batch_index += 1
            else:
                current_batch_size = N - current_index
                batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def batch_gen(self):
        while 1:
            index_array, current_index, current_batch_size = self.index_gen.__next__()
            
            img = [self.img_npy[i] for i in index_array]
            img = np.asarray(img)
            lab = [self.label_npy[i] for i in index_array]
            lab = np.asarray(lab)
            if keras.backend.image_dim_ordering() == 'th':
                nimg, ch, h, w = img.shape[0], img.shape[3], img.shape[1], img.shape[2] 
                img = np.rollaxis(img, 2, 1).reshape(nimg, ch, h, w)

            #Convert to float as most pretrained models are trained on this
            #img = img.astype('float32')
            #img /= 255
            yield img,lab



