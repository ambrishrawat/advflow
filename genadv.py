#!/usr/bin/env python

from model_defs import *
from utils import *
import argparse
import pandas as pd
import os
import numpy as np
import csv
from keras.models import model_from_json
from adv_utils import fgsm_generator, sgsm_generator

def run(csv_location,batch_size,mid,epsilon=epsilon,savedir=savedir):
     
 
    #define the batch generator (validation set)
    val_datagen = CSVGenerator(csv_location=csv_location,
                                 batch_size=batch_size)
    
    val_generator = val_datagen.batch_gen()

    # load json and create model

    json_file = open('models/'+mid+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("models/"+mid+".h5")
    print("Loaded model from disk")
 
    # compile the loaded model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    fgsm_generator(model=model, generator=val_generator)

    pass


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate adversarial images for the validation set of tinyImageNet')
    parser.add_argument('--csvpath', type=str, default='preprocessing/valset.csv', help='batch size')
    parser.add_argument('--mid', type=str, default='m1', help='model id for saving')
    parser.add_argument('--batchsize', type=str, default='32', help='batch size')
    parser.add_argument('--epsilon', type=str, default='0.3', help='epsilon for FastGradientSign method')
    parsed.add_argument('--savedir', type=str, help='location for saving the adversarial images')
    args = parser.parse_args()
    
    csv_location = args.csvpath
    mid = args.mid
    epsilon = float(args.epsilon)
    savedir = None
    batch_size = int(args.batchsize)
    if args.savedir is not None:
        savedir = args.savedir
    
    #run
    run(csv_location=csv_location,mid=mid, epsilon=epsilon,savedir=savedir)
    
    
