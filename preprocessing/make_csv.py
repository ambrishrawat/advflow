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
        traincsv_file = Path('trainset.csv')
        if traincsv_file.is_file():
            pass
        else:
            with open('trainset.csv', 'wt') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(['filename']+self.classes) #csv header
    
                c_idx = 0
                for class_ in self.classes:
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
        valcsv_file = Path('val.csv')
        if valcsv_file.is_file():
            pass
        else:
            with open(os.path.join(self.fpath,'val/val_annotations.txt'), 'rt') as csvfile:
                
                reader = csv.reader(csvfile, delimiter='\t')
                
                with open('valset.csv', 'wt') as csvfile:
                    
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Make csv files for train, val and test sets of tinyImageNet')
    parser.add_argument('--fpath', type=str, default='/dccstor/dlw/ambrish/data/tinyImageNet/tiny-imagenet-200/', help='location of the dataset')
    args = parser.parse_args()

    fpath = args.fpath

    tset = tinyImageNet(fpath)
    tset.make_csvs()
