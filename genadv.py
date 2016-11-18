#!/usr/bin/env python

from model_defs import *
from utils import *
import argparse
import pandas as pd
import os
import numpy as np
import csv
from keras.models import model_from_json
from adv_utils import *
from keras.models import load_model

     
def run(specs):


    '''Load model and weights together'''
    model = load_model('models/'+specs['save_id']+'/model.hdf5')

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    ''' load dataset, define generators'''
    c = Cifar_npy_gen(batch_size=specs['batch_size'])

 
    #fgsm_generator(model=model, generator=val_generator, nbsamples=nbsamples,
    #               epsilon=epsilon,savedir=savedir,sess=sess)

    stochastic_prediction(model=model, generator=val_generator, nbsamples=nbsamples,
                   num_feed_forwards=1000,savedir=savedir,sess=sess)
    pass


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate adversarial images and save the numpy arrays')
    parser.add_argument('--epsilon', type=str, default='0.3', help='epsilon for FastGradientSign method')
    parser.add_argument('--savedir', type=str, help='location for saving the adversarial images')
    args = parser.parse_args()
    
    csv_location = args.csvpath
    mid = args.mid
    epsilon = float(args.epsilon)
    savedir = None
    batch_size = int(args.batchsize)
    nbsamples = int(args.nbsamples)
    if args.savedir is not None:
        savedir = args.savedir

    import tensorflow as tf
    sess = tf.Session()

    # It is IMPORTANT that the session is passed here, becuase the new computation graph will be added in the seession

    from keras import backend as K
    K.set_session(sess)
    
    #arguments from the parser
    batch_size = int(args.batchsize)
    mid = args.mid

    model = lenet_ipdrop
    specs = {
            'model': model,
            'batch_size': batch_size,
            'save_id': model.__name__,
            'T': 100
            } 

    #run
    run(specs)

