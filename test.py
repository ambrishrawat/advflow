#!/usr/bin/env python

from model_defs import *
from utils import *
import argparse
import pandas as pd
import os
import numpy as np
import csv
from keras.models import model_from_json
from keras.models import load_model
import keras
from adv_utils import *

def run(specs):
     
 
    '''Load model and weights together'''
    model = load_model(os.path.join(specs['work_dir'],specs['save_id'],'model.hdf5'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   

    ''' load dataset, define generators'''
    c = Cifar_npy_gen()

    '''Generator'''
    test_ = NPYGenerator(img_npy=c.X_test,
            label_npy=c.Y_test,
            batch_size=specs['batch_size'],
            nbsamples=specs['nbsamples'])
    test_gen = test_.batch_gen()

    
    ''' Standard Dropouts '''
    
    metrics_ = model.evaluate_generator(
       generator = test_gen,
       val_samples = specs['nbsamples'])
 
    print("std-dropout(acc): %.2f%%" % (metrics_[1]*100))
    
    # to rest the generator
    test_ = NPYGenerator(img_npy=c.X_test,
            label_npy=c.Y_test,
            batch_size=specs['batch_size'],
            nbsamples=specs['nbsamples'])
    test_gen = test_.batch_gen()

    ''' MC - Dropouts '''
    mc_acc = mc_dropout_eval(model=model, 
            generator= test_gen, 
            nbsamples=specs['nbsamples'], 
            num_feed_forwards=specs['T'], 
            sess=keras.backend.get_session(),
            labels=c.Y_test)


    print("mc-dropout(acc): %.2f%%" % (mc_acc*100))

    """logging"""
    logfilename = os.path.join(specs['work_dir'],specs['save_id'],'acc.txt')
    with open(logfilename, mode='w') as logfile:
        for key in sorted(list(specs.keys())):
            value = specs[key]
            print("{}: {}".format(key, value))
            logfile.write("{}: {}\n".format(key, value))
            logfile.flush()
        
        logfile.write("std-dropout(acc): %.2f%% \n" % (metrics_[1]*100))
        logfile.write("mc-dropout(acc): %.2f%% \n" % (mc_acc*100))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Accuracy for CIFAR test set on LeNet architectures with std_droput and mc_dropout interprations')

    parser.add_argument('--csvpath', type=str, default='preprocessing/test_cifar10.csv', help='batch size')
    parser.add_argument('--batchsize', type=str, default='50', help='batch size')
    args = parser.parse_args()
    
    #arguments from the parser
    csv_location = args.csvpath
    batch_size = int(args.batchsize)
    work_dir = '/u/ambrish/models'
    nbsamples = 10000

    model = keras_eg_alldrop
    specs = {
            'model': model,
            'batch_size': batch_size,
            'save_id': model.__name__,
            'T': 100,
            'work_dir':work_dir,
            'nbsamples':nbsamples
            } 

    #compute the accuracy for the model
    run(specs)
    '''
    
    model = lenet_norelu_alldrop
    specs = {
            'model': model,
            'batch_size': batch_size,
            'save_id': model.__name__,
            'T': 100,
            'work_dir':work_dir,
            'nbsamples':10000
            } 

    #compute the accuracy for the model
    run(specs)
    
    model = lenet_norelu_nodrop
    specs = {
            'model': model,
            'batch_size': batch_size,
            'save_id': model.__name__,
            'T': 100,
            'work_dir':work_dir,
            'nbsamples':nbsamples
            } 

    #compute the accuracy for the model
    run(specs)
    '''
