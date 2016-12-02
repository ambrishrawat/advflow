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
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard



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

    tboard = TensorBoard(log_dir=os.path.join(specs['work_dir'],specs['save_id'],'logs_tb'),
                    histogram_freq=0, write_graph=False, write_images=False)


    ''' Using CSV Generator'''
    c_train = CSVGenerator(csv_location = 'train_tinyImageNet.csv',
            batch_size = specs['batch_size'])
    train_gen = c_train.batch_gen()
    
    c_test = CSVGenerator(csv_location = 'val_tinyImageNet.csv',
            batch_size = specs['batch_size'])
    test_gen = c_test.batch_gen()


    ''' Using keras Dataflow generators '''
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
        vertical_flip=True)


    ''' save the final model'''
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(specs['work_dir'],specs['save_id'],"model_arch.json"), "w") as json_file:
        json_file.write(model_json)
    

    '''call fit_generartor'''
    model.fit_generator(
        #generator=train_gen,
        generator = datagen.flow_from_directory('/dccstor/dlw/data/tinyImageNet/tiny-imagenet-restruct/train/',batch_size=specs['batch_size'], target_size = (64,64) ),
        samples_per_epoch=100000,
        nb_epoch=epochs,
        #validation_data = test_gen,
        #nb_val_samples = c_test.N,
        #callbacks=[checkpointer,earlystopping,tboard],
        callbacks=[tboard],
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
    
    model = VGG_16
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
    
    
