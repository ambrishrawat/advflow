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
import logging
def run(specs):

    logging.basicConfig(filename=os.path.join(specs['work_dir'],specs['save_id'],'exp_towards.log'),level=logging.INFO)

    '''Load model and weights together'''
    model = load_model(os.path.join(specs['work_dir'],specs['save_id'],'model.hdf5'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    e = []
    dist_tr_e = []
    mean_e = []
    std_e = []
    var_ratio_e = []
    mc_acc_e = []
    std_acc_e = []
    mean_stddr_e = []
    stddr_preds_e = []
    save_adv_e = []
    stoch_preds_e = []
    epsilon = 0.0
    
    with open(os.path.join(specs['work_dir'],specs['save_id'],'experiment_results_tw.csv'),'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['e','mean','std','var_ratio','mc_acc','std_acc'])
    
    #tw_label = np.tile(np.arange(10),10)
    #from keras.utils import np_utils
    #predictions = np_utils.to_categorical(tw_label,10)
    adv_label = np.load('adv_label.npy')

    c = Cifar_npy_gen(batch_size=specs['batch_size'])
    adv = c.X_test
    while epsilon <= specs['epsilon'] :

        print('Yoda')
        logging.info('epsilon: %f',epsilon)

        '''Load dataset and define generators'''
        #n_img = np.load('noisy_img.npy')
        #n_label = np.zeros((100,10))
        '''Get adversarial images'''
        adv = fgsm_generator_towards(model=model, 
                generator=return_gen(adv,adv_label,batch_size=specs['batch_size']), 
                nbsamples=specs['nbsamples'],
                epsilon=0.5,
                sess=keras.backend.get_session())
   
        predictions = adv_label[0:specs['nbsamples']]
        logging.info('adversarial images generated')

        #''' Nearest in training set '''
        dist_tr_ = 0.0#nearest_in_set(adv, c.X_train)
        
        #''' MC - droput stats'''
        
        stoch_preds,means_,stds_, f_m,mc_acc = mc_dropout_stats(model=model,
                generator=return_gen(adv,predictions,batch_size=specs['batch_size']),
                nbsamples=specs['nbsamples'],
                num_feed_forwards=specs['T'],
                sess=keras.backend.get_session(),
                labels=predictions)

        #update variation ratio
        f_m = 1.0 - f_m/specs['T']
        
        logging.info('mc-dropout stats computed')
        
        #''' Std - dropout stats '''
        
        mean_stddr, stddr_pred, std_acc = std_dropout_stats(model=model,
                generator=return_gen(adv,predictions,batch_size=specs['batch_size']),
                nbsamples=specs['nbsamples'],
                sess=keras.backend.get_session(),
                labels=predictions)
        logging.info('std-dropout stats computed')
        
        '''
        with open(os.path.join(specs['work_dir'],specs['save_id'],'experiment_results.csv'),'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([epsilon,
                np.mean(dist_tr_),
                np.mean(means_),
                np.mean(stds_),
                np.mean(f_m),
                mc_acc,
                np.mean(mean_stddr),
                std_acc])
        
        ''' 
        
        #append arrays
        dist_tr_e.append(np.mean(dist_tr_))
        mean_e.append(np.mean(means_))
        std_e.append(np.mean(stds_))
        var_ratio_e.append(np.mean(f_m)) 
        mc_acc_e.append(mc_acc)
        std_acc_e.append(std_acc)
        mean_stddr_e.append(np.mean(mean_stddr))
        stoch_preds_e.append(stoch_preds)
        stddr_preds_e.append(stddr_pred)
        e.append(epsilon)
    
        #save appended arrays
        np.save(os.path.join(specs['work_dir'],specs['save_id'],'stmean_e'),mean_e)
        np.save(os.path.join(specs['work_dir'],specs['save_id'],'ststd_e'),std_e)
        np.save(os.path.join(specs['work_dir'],specs['save_id'],'stvar_ratio_e'),var_ratio_e)
        np.save(os.path.join(specs['work_dir'],specs['save_id'],'stmc_acc_e'),mc_acc_e)
        np.save(os.path.join(specs['work_dir'],specs['save_id'],'ststd_acc_e'),std_acc_e)
        np.save(os.path.join(specs['work_dir'],specs['save_id'],'stmean_stddr_e'),mean_stddr_e)
        np.save(os.path.join(specs['work_dir'],specs['save_id'],'ste'),e)
        np.save(os.path.join(specs['work_dir'],specs['save_id'],'stdist_tr_e'),dist_tr_e)
        np.save(os.path.join(specs['work_dir'],specs['save_id'],'ststoch_preds_e'),stoch_preds_e)
        np.save(os.path.join(specs['work_dir'],specs['save_id'],'ststddr_preds_e'),stddr_preds_e)
        
        #TODO: save after every 5 iterations
        save_adv_e.append(adv[0:15])
        np.save(os.path.join(specs['work_dir'],specs['save_id'],'stsave_adv_e'),save_adv_e)

        epsilon += 0.002

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generate adversarial images and save the numpy arrays')
    parser.add_argument('--epsilon', type=str, default='0.1', help='epsilon for FastGradientSign method')
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
            'T':100,
            'work_dir':'/u/ambrish/models'
            } 

    #run
    run(specs)

