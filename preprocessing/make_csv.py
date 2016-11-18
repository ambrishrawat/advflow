#!/usr/bin/env python
from os import listdir
import csv
import os
import numpy as np
import pandas as pd
from scipy.misc import imresize, imread, imshow
import skimage
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import _pickle as cPickle
from scipy.misc import imsave

''' Class for preprocessing tinyImageNet dataset '''

class tinyImageNet(object):

    def __init__(self,fpath=None):
        self.fpath = fpath
        self.classes = None

    def make_train_csv(self):
        '''
        for tiny ImageNet folder structure (train)
        '''
        self.classes = listdir(os.path.join(self.fpath,'train'))
        num_classes = len(self.classes)
        traincsv_file = Path('train_tinyImageNet.csv')
        if traincsv_file.is_file():
            pass
        else:
            with open('train_tinyImageNet.csv', 'wt') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(['filename']+self.classes) #csv header
    
                c_idx = 0
                for class_ in self.classes:
                    '''
                    BUG!! - if you iterate by classes then, the corresponsing training will be biased
                    '''
                    class_path = os.path.join(self.fpath,'train',class_,'images')
                    images_ = listdir(class_path)
                    for image_ in images_:
    
                        file_path = os.path.join(class_path,image_)
    
                        label = np.zeros(num_classes)
                        label[c_idx] = 1
    
                        csvwriter.writerow([file_path] + list(map(str, label)))
                    c_idx+=1

    def make_val_csv(self):
        '''
        for tiny ImageNet folder structure (val)
        '''
        valcsv_file = Path('val_tinyImageNet.csv')
        if valcsv_file.is_file():
            pass
        else:
            with open(os.path.join(self.fpath,'val/val_annotations.txt'), 'rt') as csvfile:
                
                reader = csv.reader(csvfile, delimiter='\t')
                
                with open('val_tinyImageNet.csv', 'wt') as csvfile:
                    
                    csvwriter = csv.writer(csvfile, delimiter=',')
                    csvwriter.writerow(['filename']+self.classes)
                    
                    for row in reader:

                        file_path = os.path.join(self.fpath,'val','images',row[0])
                        
                        label = np.zeros(len(self.classes))
                        label[self.classes.index(row[1])] = 1
                        
                        csvwriter.writerow([file_path] + list(map(str, label)))            
                
    def make_csvs(self):
        self.make_train_csv()
        self.make_val_csv()



''' Global Variables, utility functions and class defintion for CIFAR raw  dataset '''

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10

# Number of files for the training-set.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def save_figs(dest_path=None, mode=None, cls=None, filenames=None, images=None):
    '''helper function to save images'''
    if mode=='train':
        for i in range(filenames.shape[0]):
            imsave(os.path.join(dest_path,mode,classes[cls[i]],filenames[i].decode('ascii')),images[i])
    else:
        for i in range(filenames.shape[0]):
            imsave(os.path.join(dest_path,mode,'images',filenames[i].decode('ascii')),images[i])
            with open(os.path.join(dest_path,mode,'annotations.csv'), 'a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow([filenames[i].decode('ascii'),classes[cls[i]]]) #csv header
            # save this in annotations classes[cls[i]]
        
        
    pass

def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images

def _unpickle(file):
    '''Unpickle the files (use bytes encoding for compatibility with Python3)'''
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo,encoding='bytes')
        return dict

class cifar10(object):
    '''
    Preprocessing the data set and making AdvFlow compatible csv files for CIFAR10
    Also, saves the JPEGs for the trianing and test set which facilitates
    easy visualisation of adversarial images
    '''
    
    
    def __init__(self,source_path = None,dest_path=None):

        #fpath is used to maintainthe destiation path where the csvfiles will be stored
        self.fpath = dest_path

        if self.fpath is None:
            raise ValueError
        self.classes = 10
        #self.save_jpegs(source_path=source_path)
    
    def save_jpegs(self,source_path=None):
        
        ''' saves JPEGs structured into class wise sub folders'''
        
        dest_path = self.fpath
        if source_path is None:
            raise ValueError
            
        files = os.listdir(source_path)
        batches = ['data_batch_1','data_batch_2', 'data_batch_3','data_batch_4', 'data_batch_5', 'test_batch']
        
        #train_batches = batches[0:5]
        #test_batches = batches[5:6]
        
        # (crude) sanity check to see if all the batch files are present or not
        from functools import reduce
        test = reduce(lambda x,y : x and y, [b in files for b in batches])
        if test is False:
            raise ValueError
            
        #make class dirs
        if not(os.path.exists(dest_path)):
            os.makedirs(dest_path)
        
        for c in classes:
            if not(os.path.exists(os.path.join(dest_path,'train',c))):
                os.makedirs(os.path.join(dest_path,'train',c))
        if not(os.path.exists(os.path.join(dest_path,'test','images'))):
            os.makedirs(os.path.join(dest_path,'test','images'))
        

        for b in batches:
            
            # Load the pickled data-file.
            data = _unpickle(source_path+'/'+b)
            
            print(data.keys())

            # Get the raw images.
            raw_images = data[b'data']
        
            # Get the class-numbers for each image. Convert to numpy-array.
            cls = np.array(data[b'labels'])

            #filenames
            filenames = np.array(data[b'filenames'])

            # Convert the images.
            images = _convert_images(raw_images)
            print(images.shape)
            
            # save images as jpegs
            if b == 'test_batch':
                save_figs(dest_path=dest_path, mode='test',cls=cls, filenames=filenames, images=images)
            else:
                save_figs(dest_path=dest_path, mode='train',cls=cls, filenames=filenames, images=images)
            

    def make_train_csv(self, traincsv=None):
        '''
        for cifar10_sets folder structure (train)
        '''
        self.classes = classes
        num_classes = len(self.classes)
        
        traincsv_file = Path(traincsv)
        if False:
            pass
        else:
            with open(traincsv, 'wt') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(['filename']+self.classes) #csv header
    
                c_idx = 0
                for class_ in self.classes:
                    class_path = os.path.join(self.fpath,'train',class_)
                    images_ = listdir(class_path)
                    for image_ in images_:
    
                        file_path = os.path.join(class_path,image_)
    
                        label = np.zeros(num_classes)
                        label[c_idx] = 1
    
                        csvwriter.writerow([file_path] + list(map(str, label)))
                    c_idx+=1

    def make_test_csv(self, valcsv=None):
        '''
        for tiny ImageNet folder structure (val)
        '''
        valcsv_file = Path(valcsv)
        if valcsv_file.is_file():
            pass
        else:
            with open(os.path.join(self.fpath,'test','annotations.csv'), 'rt') as csvfile:
                
                reader = csv.reader(csvfile, delimiter=',')
                
                with open(valcsv, 'wt') as csvfile:
                    
                    csvwriter = csv.writer(csvfile, delimiter=',')
                    csvwriter.writerow(['filename']+self.classes)
                    
                    for row in reader:

                        file_path = os.path.join(self.fpath,'test','images',row[0])
                        
                        label = np.zeros(len(self.classes))
                        label[self.classes.index(row[1])] = 1
                        
                        csvwriter.writerow([file_path] + list(map(str, label)))            
                
    def make_csvs(self):
        self.make_train_csv(traincsv='train_cifar10.csv')
        self.make_test_csv(valcsv='test_cifar10.csv')



''' Class defintion for preprocessing keras loaded CIFAR data set'''

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json

batch_size = 32
nb_classes = 10
nb_epoch = 200

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3


def save_figs2(dest_path=None, mode=None, cls=None, filenames=None, images=None):
    '''helper function to save images'''
    if mode=='train':
        for i in range(filenames.shape[0]):
            imsave(os.path.join(dest_path,mode,classes[cls[i]],filenames[i]),images[i])
    else:
        for i in range(filenames.shape[0]):
            imsave(os.path.join(dest_path,mode,'images',filenames[i]),images[i])
            with open(os.path.join(dest_path,mode,'annotations.csv'), 'a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow([filenames[i],classes[cls[i]]]) #csv header
            # save this in annotations classes[cls[i]]


    pass


class cifar10_keras(object):
    '''
    Preprocessing the data set and making AdvFlow compatible csv files for CIFAR10
    Also, saves the JPEGs for the trianing and test set which facilitates
    easy visualisation of adversarial images
    '''
    
    
    def __init__(self, dest_path = None):
        
        
        #fpath is used to maintainthe destiation path where the csvfiles will be stored
        self.fpath = dest_path

        if self.fpath is None:
            raise ValueError

        # the data, shuffled and split between train and test sets
        (self.X_train, self.y_train), (self.X_test, self.y_test) = cifar10.load_data()
        print('X_train shape:', self.X_train.shape)
        print(self.y_train.shape, 'train samples')
        print(self.y_test.shape, 'test samples')

        # convert class vectors to binary class matrices
        self.Y_train = np_utils.to_categorical(self.y_train, nb_classes)
        self.Y_test = np_utils.to_categorical(self.y_test, nb_classes)

        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        self.X_train /= 255
        self.X_test /= 255


        self.classes = 10
        #self.save_jpegs()
    
    def save_jpegs(self,source_path=None):
        
        ''' saves JPEGs structured into class wise sub folders'''
        
        dest_path = self.fpath
             
        #make class dirs
        if not(os.path.exists(dest_path)):
            os.makedirs(dest_path)
        
        for c in classes:
            if not(os.path.exists(os.path.join(dest_path,'train',c))):
                os.makedirs(os.path.join(dest_path,'train',c))
        if not(os.path.exists(os.path.join(dest_path,'test','images'))):
            os.makedirs(os.path.join(dest_path,'test','images'))
        

        # Get the raw images.
        raw_images = self.X_train
    
        # Get the class-numbers for each image. Convert to numpy-array.
        cls = self.y_train

        #filenames
        filenames = np.arange(self.X_train.shape[0])
        filenames = np.array(list(map(lambda x: str(x) + '.png',filenames)))
        # Convert the images.
        #images = _convert_images(raw_images)
        #print(images.shape)
        
        # save images as jpegs
        save_figs2(dest_path=dest_path, mode='train',cls=cls, filenames=filenames, images=self.X_train)
        

    
        # Get the raw images.
        raw_images = self.X_test
    
        # Get the class-numbers for each image. Convert to numpy-array.
        cls = self.y_test

        #filenames
        filenames = np.arange(self.X_test.shape[0])
        filenames = np.array(list(map(lambda x: str(x) + '.png',filenames)))
        # Convert the images.
        #images = _convert_images(raw_images)
        #print(images.shape)
        
        # save images as jpegs
        save_figs2(dest_path=dest_path, mode='test',cls=cls, filenames=filenames, images=self.X_test)

    def make_train_csv(self, traincsv=None):
        '''
        for cifar10_sets folder structure (train)
        '''
        self.classes = classes
        num_classes = len(self.classes)
        
        traincsv_file = Path(traincsv)
        if False:
            pass
        else:
            with open(traincsv, 'wt') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(['filename']+self.classes) #csv header
    
                c_idx = 0
                for class_ in self.classes:
                    class_path = os.path.join(self.fpath,'train',class_)
                    images_ = listdir(class_path)
                    for image_ in images_:
    
                        file_path = os.path.join(class_path,image_)
    
                        label = np.zeros(num_classes)
                        label[c_idx] = 1
    
                        csvwriter.writerow([file_path] + list(map(str, label)))
                    c_idx+=1

    def make_test_csv(self, valcsv=None):
        '''
        for tiny ImageNet folder structure (val)
        '''
        valcsv_file = Path(valcsv)
        if valcsv_file.is_file():
            pass
        else:
            with open(os.path.join(self.fpath,'test','annotations.csv'), 'rt') as csvfile:
                
                reader = csv.reader(csvfile, delimiter=',')
                
                with open(valcsv, 'wt') as csvfile:
                    
                    csvwriter = csv.writer(csvfile, delimiter=',')
                    csvwriter.writerow(['filename']+self.classes)
                    
                    for row in reader:

                        file_path = os.path.join(self.fpath,'test','images',row[0])
                        
                        label = np.zeros(len(self.classes))
                        label[self.classes.index(row[1])] = 1
                        
                        csvwriter.writerow([file_path] + list(map(str, label)))            
                
    def make_csvs(self):
        self.make_train_csv(traincsv='train_cifar10_keras.csv')
        self.make_test_csv(valcsv='test_cifar10_keras.csv')




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Make csv files for train, val and test sets of tinyImageNet')
    #parser.add_argument('--fpath', type=str, default='/dccstor/dlw/ambrish/data/tinyImageNet/tiny-imagenet-200/', help='location of the downloaded dataset (cifar10 or tinyImageNet)')
    parser.add_argument('--fpath', type=str, default='/dccstor/dlw/ambrish/data/cifar10/cifar-10-batches-py/', help='location of the downloaded dataset (cifar10 or tinyImageNet)')
    parser.add_argument('--cifarpath', type=str, default='/dccstor/dlw/ambrish/data/cifar10/cifar10-keras-sets/', help='location of the saved cifar jpegs')
    args = parser.parse_args()

    fpath = args.fpath
    cifarpath = args.cifarpath

    #tset = tinyImageNet(fpath)
    #tset = cifar10(source_path=fpath,dest_path=cifarpath)
    tset = cifar10_keras(dest_path=cifarpath)

    tset.make_csvs()
