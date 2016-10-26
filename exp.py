#!/usr/bin/env python

from marchs import *
import argparse
import pandas as pd
import os
import numpy as np
import csv

def run(epochs,batch_size):
     
    #define the optimiser and compile
    model = vgg19()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
 
    #define the batch generator    
    train_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        '/dccstor/dlw/ambrish/data/tinyImageNet/tiny-imagenet-200/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical') 


    #callbacks
    
    #call fit_generartor
    model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        nb_epoch=epochs,
        max_q_size=5)
    
    
    pass


def append_line(batch_size,t):
    r = np.concatenate([np.asarray([ncpu,ngpu,batch_size]),t])
    resultFile = open('results.csv','a')
    wr = csv.writer(resultFile,lineterminator='\n')
    wr.writerows([r])
    resultFile.close()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train VGG19 on tinyimagenet using keras')
    parser.add_argument('--epochs', type=str, default='5', help='number of epochs (the program runs through the whole data set)')
    parser.add_argument('--batchsize', type=str, default='50', help='batch size')
    args = parser.parse_args()
    
    epochs = int(args.epochs)
    batch_size = int(args.batchsize)
    
    #run the model
    run(epochs=epochs,batch_size=batch_size)
    
    #append a line of stats to the file
    #ppend_line(ncpu=ncpu,ngpu=ngpu,batch_size=batch_size,t=t)
    
    
