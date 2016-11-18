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
    directory = 'models/'+specs['save_id']
    if not os.path.exists(directory):
        os.makedirs(directory)

    """logging"""
    logfilename = "models/" + specs['save_id'] + "/log.txt"
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
            filepath='models/'+specs['save_id']+'/model.hdf5', 
            verbose=1, 
            save_best_only=True)

    earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

    c = Cifar_npy_gen(batch_size=specs['batch_size'])

    ''' save the final model'''
    # serialize model to JSON
    model_json = model.to_json()
    with open("models/"+specs['save_id']+"/model_arch.json", "w") as json_file:
        json_file.write(model_json)
    

    '''call fit_generartor'''
    model.fit_generator(
        generator=c.train_gen,
        samples_per_epoch=50000,
        nb_epoch=epochs,
        validation_data = c.test_gen,
        nb_val_samples = 10000,
        callbacks=[checkpointer, earlystopping],
        verbose=1)
 
    pass


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train a model using keras')

    #epochs, batch_size and model ID
    parser.add_argument('--epochs', type=str, default='100', help='number of epochs (the program runs through the whole data set)')
    parser.add_argument('--batchsize', type=str, default='64', help='batch size')
    parser.add_argument('--mid', type=str, default='m1', help='model id for saving')
    args = parser.parse_args()
   
    
    #arguments from the parser
    epochs = int(args.epochs)
    batch_size = int(args.batchsize)
    mid = args.mid
    
    model = lenet_alldrop
    specs = {
            'model': model,
            'epochs': epochs,
            'batch_size': batch_size,
            'save_id': model.__name__,
            'optimisation': 'adam'
            }

            

    #run the model
    run(specs)
    
    
