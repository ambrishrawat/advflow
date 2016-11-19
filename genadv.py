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
import keras
     
def run(specs):

    '''Load model and weights together'''
    model = load_model('models/'+specs['save_id']+'/model.hdf5')

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    '''Load dataset and define generators'''
    c = Cifar_npy_gen(batch_size=specs['batch_size'])

    '''
    metrics_ = model.evaluate_generator(
           generator = return_gen(c.X_test,c.Y_test,batch_size=specs['batch_size']),
           val_samples = 10000)

    print("(Before) std-dropout(acc): %.2f%%" % (metrics_[1]*100))
    '''

    '''Get adversarial images'''
    adv, predictions = fgsm_generator(model=model, 
            generator=return_gen(c.X_test,c.Y_test,batch_size=specs['batch_size']), 
            nbsamples=specs['nbsamples'],
            epsilon=specs['epsilon'],
            sess=keras.backend.get_session())
   

    save_npy(np_array = c.X_test, 
            specs = {'work_dir': '/u/ambrish/nparrays', 
                'save_id': 'adv'+str(specs['epsilon']),
                'file_id': 'orig_img'})
    
    save_npy(np_array = adv, 
            specs = {'work_dir': '/u/ambrish/nparrays', 
                'save_id': 'adv'+str(specs['epsilon']),
                'file_id': 'adv_img'})
 
    save_npy(np_array = predictions, 
            specs = {'work_dir': '/u/ambrish/nparrays', 
                'save_id': 'adv'+str(specs['epsilon']),
                'file_id': 'label'})
    
    metrics_ = model.evaluate_generator(
           generator = return_gen(c.X_test,predictions,batch_size=specs['batch_size']),
           val_samples = 10000)

    print("(Before) std-dropout(acc): %.2f%%" % (metrics_[1]*100))

    metrics_ = model.evaluate_generator(
           generator = return_gen(adv,predictions,batch_size=specs['batch_size']),
           val_samples = 10000)

    print("(After) std-dropout(acc): %.2f%%" % (metrics_[1]*100))
   
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate adversarial images and save the numpy arrays')
    parser.add_argument('--epsilon', type=str, default='0.001', help='epsilon for FastGradientSign method')
    parser.add_argument('--savedir', type=str, help='location for saving the adversarial images')
    args = parser.parse_args()
    
    #arguments from the parser
    epsilon = float(args.epsilon)
    savedir = None
    if args.savedir is not None:
        savedir = args.savedir


    # It is IMPORTANT that the session is passed here, becuase the new computation graph will be added in the seession
    

    model = lenet_ipdrop
    specs = {
            'batch_size': 64,
            'save_id': model.__name__,
            'nbsamples':10000,
            'epsilon':epsilon
            } 

    #run
    run(specs)

