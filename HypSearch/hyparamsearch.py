from __future__ import print_function
from __future__ import division
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
from torchviz import make_dot, make_dot_from_trace

from sklearn.metrics import roc_auc_score  
from sklearn.metrics import roc_curve 

#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle


from bayes_opt import BayesianOptimization
import model_RNN as model
import main_HPS as main_run
use_cuda = torch.cuda.is_available()


def print2file(buf, outFile):
	outfd = open(outFile, 'a')
	outfd.write(buf + '\n')
	outfd.close()

logFile='testRNN1.log'
header = 'Model|EmbSize|CellType|n_Layers|Hidden|Dropout|Optimizer|LR|L2|EPs|BestValidAUC|TestAUC|atEpoch'
print2file(header, logFile)

def model_tune(dlm_code, embdim_exp, hid_exp, layers_n, dropout, ct_code , opt_code , l2_exp , lr_exp , eps_exp, bt_size):
    
    embed_dim = 2** int(embdim_exp)
    hidden_size = 2** int(hid_exp)
    n_layers = int(layers_n)
    dropout = round(dropout,1)
    
    if int(dlm_code)<3:
      if int(ct_code) ==1:
          cell_type='RNN'   
      elif int(ct_code) ==2:
          cell_type='LSTM'
      elif int(ct_code) ==3:
          cell_type='GRU'
      
    if int(dlm_code)==1:
        w_model='RNN'
        ehr_model = model.EHR_RNN(17000, embed_dim, hidden_size, n_layers, dropout, cell_type)
    elif int(dlm_code)==2:
        w_model='DRNN'
        ehr_model = model.DRNN(17000, embed_dim, hidden_size, n_layers, dropout, cell_type)
      #elif int(dlm_code)==3:
      #  w_model='CNN'
      #  ehr_model = model.EHR_CNN(17000, embed_dim, hidden_size, n_layers, dropout, cell_type)
        
    if use_cuda:
        ehr_model = ehr_model.cuda()

    l2 = 10** int(l2_exp)
    lr = 10** int(lr_exp)
    eps = 10** int(eps_exp)
    
    if int(opt_code) ==1:
        opt= 'Adadelta'
        optimizer = optim.Adadelta(ehr_model.parameters(), lr=lr, weight_decay=l2 ,eps=eps) ## rho=0.9
    elif int(opt_code) ==2:
        opt= 'Adagrad'
        optimizer = optim.Adagrad(ehr_model.parameters(), lr=lr, weight_decay=l2) ##lr_decay no eps
    elif int(opt_code) ==3:
        opt= 'Adam'
        optimizer = optim.Adam(ehr_model.parameters(), lr=lr, weight_decay=l2,eps=eps ) ## Beta defaults (0.9, 0.999), amsgrad (false)
    elif int(opt_code) ==4:
         opt= 'Adamax'
         optimizer = optim.Adamax(ehr_model.parameters(), lr=lr, weight_decay=l2 ,eps=eps) ### Beta defaults (0.9, 0.999)
    elif int(opt_code) ==5:
         opt= 'RMSprop'
         optimizer = optim.RMSprop(ehr_model.parameters(), lr=lr, weight_decay=l2 ,eps=eps)                
    elif int(opt_code) ==6:
         opt= 'ASGD'
         optimizer = optim.ASGD(ehr_model.parameters(), lr=lr, weight_decay=l2 ) ### other parameters
    elif int(opt_code) ==7:
         opt= 'SGD'
         optimizer = optim.SGD(ehr_model.parameters(), lr=lr, weight_decay=l2 ) ### other parameters
    #elif int(opt_code) ==8:
    #     opt= 'SparseAdam'
    #     optimizer = optim.SparseAdam(ehr_model.parameters(), lr=lr , eps=eps) ### Beta defaults (0.9, 0.999) no weight_decay

               
    batch_size= int(bt_size)
    bestValidAuc, buf = main_run.model_run (25,ehr_model,optimizer,batch_size,w_model)
    pFile= w_model+'|'+str(embed_dim)+'|'+cell_type+'|'+str(n_layers)+'|'+str(hidden_size)+'|'+str(dropout)+'|'+opt+'|'+str(lr)+'|'+str(l2)+'|'+str(eps)+ buf    
    #little transformations to use the searched values
    
    print2file(pFile, logFile)
 
    return bestValidAuc
    

if __name__ == "__main__":
    gp_params = {"alpha": 1e-4}

    LRBO = BayesianOptimization(model_tune,
        {'dlm_code': (1, 2),'embdim_exp': (5, 9),'hid_exp': (4, 8),'layers_n': (1, 5),'dropout': (0, 1),'ct_code': (1, 3),'opt_code': (1, 7),'l2_exp': (-5, -1), 'lr_exp': (-5, -1),'eps_exp': (-9, -6),'bt_size': (200, 200)})
    LRBO.explore({'dlm_code': [2],'embdim_exp': [8],'hid_exp': [7],'layers_n': [2],'dropout': [0.3],'ct_code': [3],'opt_code': [3],'l2_exp': [-4], 'lr_exp': [-3],'eps_exp': [-7],'bt_size': [200]})

    LRBO.maximize(n_iter=50, **gp_params)

    print('-' * 53)
    print('Final Results')
    print('RNN / DRNN: %f' % LRBO.res['max']['max_val'])
    
    print2file(LRBO.res['max'], logFile)

