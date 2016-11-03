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

class ForEveryEpoch(Callback):

    def __init__(self,valcsvpath=None,mid=None):
        
        self.mid = mid
        self.val_datagen = CSVGenerator(csv_location=valcsvpath,
                                 batch_size=batch_size, target_size=(32,32))
    
        self.val_generator = self.val_datagen.batch_gen()

        self.tbeg = None
        self.tend = None

    def on_epoch_end(self, epoch, logs={}):

        ''' Save after every 100 epoch'''

        if epoch%20==0 and epoch != 0:

            # serialize weights to HDF5
            self.model.save_weights("models/"+mid+'/snap_e'+str(epoch)+".h5")
            print("Saved model to disk")


        ''' Log accuracy on validation set  after every 10 epochs '''

        if epoch%10==0:
            
            #evaluate the model on the validation set
            val_loss = self.model.evaluate_generator(generator = self.val_generator,
                                                val_samples = self.val_datagen.get_data_size())

            #print("%s: %.2f%%" % (model.metrics_names[1], val_loss[1]*100))
            
            #log the result in a csv
            with open('models/'+mid+'/acc_log.csv','a') as resultFile:
                wr = csv.writer(resultFile,lineterminator='\n',delimiter='\t')
                wr.writerow(list(map(str,[epoch,val_loss])))


        ''' Log epoch number after every epoch '''
    
        if epoch%5==0:
 
            self.tend = time.time()
            #log the result in a csv
            with open('models/'+mid+'/logs.csv','a') as resultFile:
                
                wr = csv.writer(resultFile,lineterminator='\n',delimiter='\t')
                wr.writerow(list(map(str,[epoch,self.tbeg,self.tend,self.tend-self.tbeg])))

    def on_epoch_begin(self, epoch, logs={}):
        self.tbeg = time.time()
        pass

    #def on_batch_end(self, batch, logs={}):
    #    print('Batch ends: '+str(batch))


def run(csvpath,valcsvpath,epochs,batch_size,mid):
     
    '''define the optimiser and compile'''
    #model = VGG_16_3()
    #model = VGG_16_pretrain_1()
    model = cifar_keras()

    opt = SGD(lr=0.001, decay=1.e-6, momentum=0.9, nesterov=False)
    #opt = RMSprop(lr=0.001)
    #opt = Adadelta(lr=0.001)
    #opt = Adam(lr=0.001)
    #opt = Adagrad(lr=0.001)

    opt_tag = 'sgd = SGD(lr=0.001, decay=1.e-6, momentum=0.9, nesterov=False)'    
    #opt_tag = 'rms = RMSprop(lr=0.0001)'
    #opt_tag = 'adadelta = Adadelta(lr=0.001)'
    #opt_tag = 'adam = Adam(lr=0.001)'
    #opt_tag = 'adagrad= Adagrad(lr=0.001)'

    model.compile(loss='categorical_crossentropy', optimizer=opt, metric=['accuracy'])
 
    '''define the batch generator   (training set)'''
    train_datagen = CSVGenerator(csv_location=csvpath,
                                 batch_size=batch_size,
                                 target_size=(32,32),shuffle=False)

    train_generator = train_datagen.batch_gen()
    
    '''define the batch generator   (validation set)'''
    #val_datagen = CSVGenerator(csv_location='preprocessing/valset.csv',
    #                             batch_size=batch_size)
    #
    #val_generator = val_datagen.batch_gen()


    '''callback for epochs'''
    cEpochs = ForEveryEpoch(valcsvpath=valcsvpath,mid=mid)


    # serialize model to JSON

    directory = 'models/'+mid
    if not os.path.exists(directory):
        os.makedirs(directory)

    model_json = model.to_json()
    with open("models/"+mid+"/model_arch.json", "w") as json_file:
        json_file.write(model_json)


    with open('models/'+mid+'/logs.csv','a') as resultFile:
                
         wr = csv.writer(resultFile,lineterminator='\n')
         wr.writerow(['Epoch',str(epochs)])
         wr.writerow(['Batch size', str(batch_size)])
         wr.writerow(['Optimisation',opt_tag])


    '''call fit_generartor'''
    model.fit_generator(
        generator=train_generator,
        samples_per_epoch=train_datagen.get_data_size(),
        #samples_per_epoch=,
        nb_epoch=epochs,
        callbacks=[cEpochs],
        #validation_data = val_generator,
        #nb_val_samples = val_datagen.get_data_size(),
        verbose=1)
   
    pass


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train a model using keras')
    parser.add_argument('--csvpath', type=str, default='preprocessing/train_cifar10.csv', 
                        help='csv location for the training set csv file')
    parser.add_argument('--valcsvpath', type=str, default='preprocessing/test_cifar10.csv', 
                        help='csv location for the validation set csv file')
    #parser.add_argument('--csvpath', type=str, default='preprocessing/train_tinyImageNet.csv', 
    #                    help='csv location for the training set csv file')
    #parser.add_argument('--valcsvpath', type=str, default='preprocessing/val_tinyImageNet.csv', 
    #                    help='csv location for the validation set csv file')
    parser.add_argument('--epochs', type=str, default='5', help='number of epochs (the program runs through the whole data set)')
    parser.add_argument('--batchsize', type=str, default='50', help='batch size')
    parser.add_argument('--mid', type=str, default='m1', help='model id for saving')
    args = parser.parse_args()
   
    csvpath = args.csvpath
    valcsvpath = args.valcsvpath
    epochs = int(args.epochs)
    batch_size = int(args.batchsize)
    mid = args.mid
    
    #run the model
    run(csvpath=csvpath,valcsvpath=valcsvpath,epochs=epochs,batch_size=batch_size,mid=mid)
    
    
