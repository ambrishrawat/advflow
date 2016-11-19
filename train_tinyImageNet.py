#!/usr/bin/env python

from model_defs import *
from utils import *
import argparse
import pandas as pd
import os
import numpy as np
import csv
from keras.callbacks import Callback
import time
from keras.optimizers import RMSprop, SGD, Adagrad, Adadelta, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping



def run(specs):
    
    ''' Make the mid directory '''
    directory = os.path.join(specs['work_dir'],specs['save_id'])
    if not os.path.exists(directory):
        os.makedirs(directory)

    """logging"""
    logfilename = os.path.join(specs['work_dir'],specs['save_id'],"log.txt")
    with open(logfilename, mode='w') as logfile:
        for key in sorted(list(specs.keys())):
            value = specs[key]
            print("{}: {}".format(key, value))
            logfile.write("{}: {}\n".format(key, value))
            logfile.flush()

    '''define the optimiser and compile'''
    model = specs['model']()

    model.compile(loss='categorical_crossentropy', 
            optimizer=specs['optimisation'], 
            metrics=['accuracy'])

    '''callbacks'''
    checkpointer = ModelCheckpoint(
            filepath=os.path.join(specs['work_dir'],specs['save_id'],'model.hdf5'), 
            verbose=1, 
            save_best_only=True)

    earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)


    c_train = CSVGenerator(csv_location = 'preprocessing/random.csv',
            batch_size = specs['batch_size'])
    train_gen = c_train.batch_gen()

    c_test = CSVGenerator(csv_location = 'preprocessing/val_tinyImageNet.csv',
            batch_size = specs['batch_size'])
    test_gen = c_test.batch_gen()


    ''' save the final model'''
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(specs['work_dir'],specs['save_id'],"model_arch.json"), "w") as json_file:
        json_file.write(model_json)
    

    '''call fit_generartor'''
    model.fit_generator(
        generator=train_gen,
        samples_per_epoch=c_train.N,
        nb_epoch=epochs,
        validation_data = test_gen,
        nb_val_samples = c_test.N,
        callbacks=[checkpointer,earlystopping],
        verbose=1)
 
    pass


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Training CIFAR example for LeNet architectures')

    #epochs, batch_size and model ID
    parser.add_argument('--epochs', type=str, default='200', help='number of epochs (the program runs through the whole data set)')
    parser.add_argument('--batchsize', type=str, default='64', help='batch size')
    args = parser.parse_args()
   
    
    #arguments from the parser
    epochs = int(args.epochs)
    batch_size = int(args.batchsize)
    
    model = VGG_16_pretrain_2
    specs = {
            'model': model,
            'epochs': epochs,
            'batch_size': batch_size,
            'save_id': model.__name__,
            'optimisation': 'adam',
            'work_dir': '/u/ambrish/models'
            }

            

    #run the model
    run(specs)
    
    
