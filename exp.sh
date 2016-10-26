#!/usr/bin/env python

from voc2012.voc_utils import *
from voc2012.utils import *
from voc2012.vgg19 import *#
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
    
    #get the dataframe
    df = pd.read_pickle('train.pkl')
    
    #define the batch generator    
    df_gen = DfIterator(df,batch_size=batch_size)
    train_generator = df_gen.batch_gen()
 
    #callbacks
    time_ = EpochTime()
    
    #call fit_generartor
    model.fit_generator(
        train_generator,
        samples_per_epoch=df.shape[0],
        nb_epoch=epochs,
        callbacks=[time_],
        max_q_size=5)
    
    #retrieve time_begin
    
    time_begin = np.array(time_.time_begin)
    #time_begin = np.asarray([1.32443,1.3434,1256.646,1.3545,23234.321])
    t = np.ediff1d(time_begin)
    return t


def append_line(ncpu,ngpu,batch_size,t):
    r = np.concatenate([np.asarray([ncpu,ngpu,batch_size]),t])
    resultFile = open('results.csv','a')
    wr = csv.writer(resultFile,lineterminator='\n')
    wr.writerows([r])
    resultFile.close()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train VGG19 on VOC2012 using keras')
    parser.add_argument('--ncpu', type=str, default='8', help='number of cpus')
    parser.add_argument('--ngpu', type=str, default='1', help='number of GPUs')
    parser.add_argument('--epochs', type=str, default='5', help='number of epochs (the program runs through the whole data set)')
    parser.add_argument('--batchsize', type=str, default='50', help='batch size')
    args = parser.parse_args()
    
    ncpu = int(args.ncpu)
    ngpu = int(args.ngpu)
    epochs = int(args.epochs)
    batch_size = int(args.batchsize)
    
    #run the model
    t = run(epochs=epochs,batch_size=batch_size)
    
    #append a line of stats to the file
    append_line(ncpu=ncpu,ngpu=ngpu,batch_size=batch_size,t=t)
    
    
