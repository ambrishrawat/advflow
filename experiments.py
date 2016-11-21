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
    model = load_model(os.path.join(specs['work_dir'],specs['save_id'],'model.hdf5'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    e = []
    mean_e = []
    std_e = []
    var_ratio_e = []
    mc_acc_e = []
    std_acc_e = []
    save_adv = []
    epsilon = 0.0 

    with open(os.path.join(specs['work_dir'],specs['save_id'],'experiment_results.csv'),'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['e','mean','std','var_ratio','mc_acc','std_acc'])
    
    while epsilon <=specs['epsilon'] :

        '''Load dataset and define generators'''
        c = Cifar_npy_gen(batch_size=specs['batch_size'])
        
        '''Get adversarial images'''
        adv, predictions = fgsm_generator(model=model, 
                generator=return_gen(c.X_test,c.Y_test,batch_size=specs['batch_size']), 
                nbsamples=specs['nbsamples'],
                epsilon=epsilon,
                sess=keras.backend.get_session())
      

        stoch_preds,means_,stds_, f_m = mc_dropout_stats(model=model,
                generator=return_gen(adv,predictions,batch_size=specs['batch_size']),
                nbsamples=specs['nbsamples'],
                num_feed_forwards=specs['T'],
                sess=keras.backend.get_session())


        f_m = 1.0 - f_m/specs['T']
        #compute from stoch_preds and predictions
        #stats_(stoch_preds = stoch_preds, predictions= predictions)

        ''' MC - accuracy '''
        mc_acc = mc_dropout_eval(model=model,
                generator=return_gen(adv,predictions,batch_size=specs['batch_size']),
                nbsamples=specs['nbsamples'],
                num_feed_forwards=specs['T'],
                sess=keras.backend.get_session())

        ''' Std - accuracy '''
        metrics_ = model.evaluate_generator(
               generator = return_gen(adv,predictions,batch_size=specs['batch_size']),
               val_samples = specs['nbsamples'])

        
        with open(os.path.join(specs['work_dir'],specs['save_id'],'experiment_results.csv'),'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([epsilon, 
                np.mean(means_),
                np.mean(stds_),
                np.mean(f_m),
                mc_acc,
                metrics_[1]])

        mean_e.append(np.mean(means_))
        std_e.append(np.mean(stds_))
        var_ratio_e.append(np.mean(f_m))
        mc_acc_e.append(mc_acc)
        epsilon += 0.0001
        std_acc_e.append(metrics_[1])
        save_adv.append(adv)
        e.append(epsilon)

    np.save(os.path.join(specs['work_dir'],specs['save_id'],'mean_e'),mean_e)
    np.save(os.path.join(specs['work_dir'],specs['save_id'],'std_e'),std_e)
    np.save(os.path.join(specs['work_dir'],specs['save_id'],'var_ratio_e'),var_ratio_e)
    np.save(os.path.join(specs['work_dir'],specs['save_id'],'mc_acc_e'),mc_acc_e)
    np.save(os.path.join(specs['work_dir'],specs['save_id'],'std_acc_e'),std_acc_e)
    np.save(os.path.join(specs['work_dir'],specs['save_id'],'e'),e)
    np.save(os.path.join(specs['work_dir'],specs['save_id'],'save_adv'),save_adv)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate adversarial images and save the numpy arrays')
    parser.add_argument('--epsilon', type=str, default='0.01', help='epsilon for FastGradientSign method')
    parser.add_argument('--savedir', type=str, help='location for saving the adversarial images')
    args = parser.parse_args()
    
    #arguments from the parser
    epsilon = float(args.epsilon)
    savedir = None
    if args.savedir is not None:
        savedir = args.savedir


    # It is IMPORTANT that the session is passed here, becuase the new computation graph will be added in the seession
    

    model = keras_eg_alldrop
    specs = {
            'batch_size': 200,
            'save_id': model.__name__,
            'nbsamples':10000,
            'epsilon':epsilon,
            'T':200,
            'work_dir':'/u/ambrish/models'
            } 

    #run
    run(specs)

