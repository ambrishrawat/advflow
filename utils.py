import tensorflow as tf
import os
import pandas as pd
from bs4 import BeautifulSoup
from more_itertools import unique_everseen
import numpy as np

from keras.callbacks import Callback
import time
import csv
from scipy.misc import imresize, imread, imshow
import skimage
from skimage import io
import keras

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
    img = skimage.img_as_float(imread(
        img_filename,mode='RGB')).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    #'''Subtract channel vise mean'''
    #ch_mean = np.load('preprocessing/ch_mean.npy')
    #img = img - ch_mean
    return img



class CSVGenerator():
    '''
    Class for iterating over a pandas data frame with filenames and addresses

    TODO
    add comments
    '''

    def __init__(self,csv_location = None,
                 batch_size=32,
                 shuffle=False,
                 target_size=(32,32)):
        
        if csv_location is not None:
            self.df = pd.read_csv(csv_location)
        self.N = self.df.shape[0]
        print(self.N)
        self.batch_size = batch_size
        self.total_batches_seen = 0
        self.index_gen = self._idx_gen(self.N,batch_size,shuffle)
        self.target_size = target_size

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
            img = [imresize(load_img(self.df.iloc[i]['filename']),size=self.target_size) for i in index_array]
            img = np.asarray(img)
            lab = [self.df.iloc[i].iloc[1:].values.astype('float32') for i in index_array]
            lab = np.asarray(lab)
            if keras.backend.image_dim_ordering() == 'th':
                nimg, ch, h, w = img.shape[0], img.shape[3], img.shape[1], img.shape[2] 
                img = np.rollaxis(img, 2, 1).reshape(nimg, ch, h, w)

            #print('Yoda'+str(index_array[0]))
            yield img,lab







