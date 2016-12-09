#!/usr/bin/env python

from model_defs import *
from utils import *
import argparse
import pandas as pd
import os
import numpy as np
import csv
from keras.callbacks import Callback
from keras.models import load_model
import time
from keras.optimizers import RMSprop, SGD, Adagrad, Adadelta, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


def preprocess_input(x, dim_ordering='default'):
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
    # 'RGB'->'BGR'
    x = x[:, :, :, ::-1]
    return x


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
    print('Nodel defined or loaded')
    #model = load_model(os.path.join(specs['work_dir'],specs['save_id'],'model2.hdf5'))

    sgd = SGD(lr=1.e-3, decay=0.0, momentum=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', 
            #optimizer=specs['optimisation'], 
            optimizer=sgd, 
            metrics=['accuracy'])

    '''callbacks'''
    checkpointer = ModelCheckpoint(
            filepath=os.path.join(specs['work_dir'],specs['save_id'],'model0ihdf5'), 
            verbose=1, 
            save_best_only=True)

    earlystopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

    tboard = TensorBoard(log_dir=os.path.join(specs['work_dir'],specs['save_id'],'logs_tb0'),
                    histogram_freq=0, write_graph=False, write_images=False)


    ''' Using CSV Generator'''
    c_train = CSVGenerator(csv_location = 'train_tinyImageNet.csv',
            batch_size = specs['batch_size'])
    train_gen = c_train.batch_gen()
    
    c_test = CSVGenerator(csv_location = 'val_tinyImageNet.csv',
            batch_size = specs['batch_size'])
    test_gen = c_test.batch_gen()


    ''' Using keras Dataflow generators '''
    datagen_train = ImageDataGenerator(
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

    train_gen = datagen_train.flow_from_directory('/dccstor/dlw/data/tinyImageNet/tiny-imagenet-restruct/train/',
                batch_size=specs['batch_size'], target_size = (224,224))

    def gen_train():
        while 1:
            x,y = train_gen.__next__()
            yield preprocess_input(x),y
        

    ''' Using keras Dataflow generators '''
    datagen_val = ImageDataGenerator(
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


    val_gen = datagen_val.flow_from_directory('/dccstor/dlw/data/tinyImageNet/tiny-imagenet-restruct/val/',
                batch_size=specs['batch_size'], target_size = (224,224))

    def gen_val():
        while 1:
            x,y = val_gen.__next__()
            yield preprocess_input(x),y

    ''' save the final model'''
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(specs['work_dir'],specs['save_id'],"model_arch.json"), "w") as json_file:
        json_file.write(model_json)
    

    '''call fit_generartor'''
    model.fit_generator(
        #generator=train_gen,
        generator = gen_train(),
        samples_per_epoch=100000,
        nb_epoch=epochs,
        validation_data = gen_val(),
        nb_val_samples = 10000,
        callbacks=[checkpointer,earlystopping,tboard],
        #callbacks=[tboard],
        verbose=1)

    model.save(os.path.join(specs['work_dir'],specs['save_id'],"model_final.h5"))


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
    
    model = VGG16_keras
    specs = {
            'model': model,
            'epochs': epochs,
            'batch_size': batch_size,
            'save_id': model.__name__,
            'optimisation': 'rmsprop',
            'work_dir': '/u/ambrish/models'
            }

            

    #run the model
    run(specs)
    
    
