#!/usr/bin/env python

from model_defs import *
from utils import *
import argparse
import pandas as pd
import os
import numpy as np
import csv

def run(epochs,batch_size,mid):
     
    #define the optimiser and compile
    model = vgg19()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metric=['accuracy'])
 
    #define the batch generator   (training set)
    train_datagen = CSVGenerator(csv_location='preprocessing/trainset.csv',
                                 batch_size=batch_size)

    train_generator = train_datagen.batch_gen()
    
    #define the batch generator   (validation set)
    val_datagen = CSVGenerator(csv_location='preprocessing/valset.csv',
                                 batch_size=batch_size)
    
    val_generator = val_datagen.batch_gen()


    #call fit_generartor
    model.fit_generator(
        generator=train_generator,
        samples_per_epoch=train_datagen.get_data_size(),
        #samples_per_epoch=batch_size,
        nb_epoch=epochs,
        validation_data = val_generator,
        nb_val_samples = val_datagen.get_data_size(),
        verbose=2)
   

    # serialize model to JSON
    model_json = model.to_json()
    with open("models/"+mid+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("models/"+mid+".h5")
    print("Saved model to disk")

    pass


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train VGG19 on tinyimagenet using keras')
    parser.add_argument('--epochs', type=str, default='5', help='number of epochs (the program runs through the whole data set)')
    parser.add_argument('--batchsize', type=str, default='50', help='batch size')
    parser.add_argument('--mid', type=str, default='m1', help='model id for saving')
    args = parser.parse_args()
    
    epochs = int(args.epochs)
    batch_size = int(args.batchsize)
    mid = args.mid
    
    #run the model
    run(epochs=epochs,batch_size=batch_size,mid=mid)
    
    
