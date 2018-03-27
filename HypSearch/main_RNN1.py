# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 09:25:59 2018

@author: jzhu8
"""
from __future__ import print_function, division
from io import open
import string
import re
import random

import os
import sys
import argparse
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
#from torchviz import make_dot, make_dot_from_trace

from sklearn.metrics import roc_auc_score  
from sklearn.metrics import roc_curve 

#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle
    
#import self-defined modules
import model_RNN as model # we have all models in this file
import Loaddata
import TrVaTe as TVT #as TrainVaTe

# check GPU availability
use_cuda = torch.cuda.is_available()

print (use_cuda )
print ( torch.cuda.current_device())

#torch.cuda.set_device(1)
#print (use_cuda )
#print ( torch.cuda.current_device())

'''
parser = argparse.ArgumentParser(description='EHR HF prediction with Pytorch: LR, RNN, CNN')
# learning
parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.0001]')
parser.add_argument('-L2', type=float, default=0, help='L2 regularization [default: 0]')
parser.add_argument('-epochs', type=int, default=5, help='number of epochs for train [default: 5]')
parser.add_argument('-batch_size', type=int, default=200, help='batch size for training [default: 200]')
#parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
#parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-seq_file', type = str, default = 'Data/h143.visits' , help='the path to the Pickled file containing visit information of patients')
parser.add_argument('-label_file', type = str, default = 'Data/h143.labels', help='the path to the Pickled file containing label information of patients')
parser.add_argument('-validation_ratio', type = float, default = 0.1, help='validation data size [default: 0.1]')
parser.add_argument('-test_ratio', type = float, default = 0.2, help='test data size [default: 0.2]')
# model
parser.add_argument('-which_model', type = str, default = 'RNN', help='choose from {"LR", "RNN", "DRNN", "CNN"}')
#parser.add_argument('-mb', type = bool, default =  True , help='whether train on mini batch (True) or not (False) [default: False ]') #at train
parser.add_argument('-input_size', type = int, default =20000, help='input dimension [default: 20000]')
parser.add_argument('-embed_dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-hidden_size', type=int, default=128, help='size of hidden layers [default: 128]')
parser.add_argument('-ch_out', type=int, default=64, help='number of each kind of kernel [default; 64]')
parser.add_argument('-kernel_sizes', type=list, default=[3], help='comma-separated kernel size to use for convolution [default:[3]')
parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.1]')
parser.add_argument('-n_layers', type=int, default=1, help='number of Layers, for dilated models, dilations will increase exponentialy with mumber of layers [default: 1]')
parser.add_argument('-cell_type', type = str, default = 'GRU', help='For RNN based models choose from {"RNN", "GRU", "LSTM"}')
parser.add_argument('-eb_mode', type=str, default='sum', help= "embedding mode [default: 'sum']")

# option
#parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
args = parser.parse_args()
'''

# load and prepare data
set_x = pickle.load(open('Data/h143.visits', 'rb'), encoding='bytes')
set_y = pickle.load(open('Data/h143.labels', 'rb'),encoding='bytes')

'''
#preprocessing
# LR needs to have input format of list; list of list for NN models
if args.which_model == 'LR':
    model_x = []
    for patient in set_x:
        model_x.append([each for visit in patient for each in visit])  
else: 
    model_x = set_x     
'''
    
merged_set= [[set_y[i],set_x[i]] for i in range(len(set_y))] #list of list or list of lists of lists
print("\nLoading and preparing data...")    
train1, valid1, test1 = Loaddata.load_data(merged_set)
#print("\nSample data after split:")  
#print(train1[0])

'''
# model loading part: choose which model to use 
if args.which_model == 'RNN':
    ehr_model = model.EHR_RNN(args.input_size, args.embed_dim, args.hidden_size, args.n_layers, args.dropout, args.cell_type) 

elif args.which_model == 'DRNN':    
    ehr_model = model.DRNN(args.input_size, args.embed_dim, args.hidden_size, args.n_layers, args.dropout, args.cell_type) 

else: 
    print ('model can be either RNN or DRNN')

if use_cuda:
    ehr_model = ehr_model.cuda()
    
#outFile='Output/'+ args.which_model+'_'+str(args.embed_dim)+'_'+args.cell_type+'_L'+str(args.n_layers)+'_H'+str(args.hidden_size)+'_D'+str(args.dropout)+'_LR'+str(args.lr)+'_P'+str(args.L2)
#print ("output file name: ",outFile)


#optimizer = optim.Adam(ehr_model.parameters(), lr=args.lr, weight_decay=args.L2)

'''
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)




def model_run(epochs, ehr_model,optimizer,batch_size,w_model ):
  ## train validation and test part
  #epochs=args.epochs
  #batch_size=args.batch_size
  
  current_loss_allep=[]
  all_losses_allep=[]
  avg_losses_allep=[]
  #train_auc_allep =[]
  valid_auc_allep =[]
  #test_auc_allep=[]
  bestValidAuc = 0.0
  bestTestAuc = 0.0
  bestValidEpoch = 0
    
  # train, validation, and test for each epoch 
  
  for ep in range(epochs):
      start = time.time()
      current_loss, train_loss = TVT.train(train1, model= ehr_model, optimizer = optimizer, batch_size = batch_size)
      avg_loss = np.mean(train_loss)
      train_time = timeSince(start)
      eval_start = time.time()
      #train_auc, y_real, y_hat = TVT.calculate_auc(model= ehr_model, data = train1, which_model = args.which_model, batch_size = args.batch_size)
      valid_auc, y_real, y_hat  = TVT.calculate_auc(model = ehr_model, data = valid1, which_model = w_model, batch_size = batch_size)
      #print ("Epoch ", ep, "Summary:  Training_auc :", train_auc, " , Validation_auc : ", valid_auc, " ,& Test_auc : " , test_auc, " Avg Loss: ", avg_loss, 'Train Time (%s) Eval Time (%s)'%(train_time,eval_time) )
      #buf = 'Epoch:%d, Training_auc:%f, Validation_AUC:%f , Test_AUC:%f , Avg Loss:%f , Train_Time:%s  , Eval_Time:%s' % (ep, train_auc, valid_auc,test_auc,avg_loss,train_time,eval_time)
      #print(buf)
      #print2file(buf, logFile)		
      if valid_auc > bestValidAuc: 
          bestValidAuc = valid_auc
          bestValidEpoch = ep
          #bestTestAuc = test_auc
          bestTestAuc, y_real, y_hat = TVT.calculate_auc(model = ehr_model, data = test1, which_model = w_model, batch_size = batch_size)
      eval_time = timeSince(eval_start)      
       
      current_loss_allep.append(current_loss)
      all_losses_allep.append(train_loss)
      avg_losses_allep.append(avg_loss)
      #train_auc_allep.append(train_auc)
      valid_auc_allep.append(valid_auc)
      #test_auc_allep.append(test_auc)  
      if ep - bestValidEpoch >3:
          break
      
  buf = 'The best validation & test AUC:%f, %f at epoch:%d ' % (bestValidAuc, bestTestAuc, bestValidEpoch)


  return bestValidAuc , buf

##Saving models after the best epoch? torch.save(MyModel, './model.pth') 



#model_run (args.epochs, outFile + '.log' )