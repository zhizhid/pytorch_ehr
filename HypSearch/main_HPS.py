# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:25:59 2018

@author: lgindybekhet
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

from sklearn.metrics import roc_auc_score  
from sklearn.metrics import roc_curve 

import numpy as np

try:
    import cPickle as pickle
except:
    import pickle
    
#import self-defined modules
import model_HPS as model # we have all models in this file
import Loaddata
import TrVaTe as TVT #as TrainVaTe

# check GPU availability
use_cuda = torch.cuda.is_available()
print (use_cuda , torch.cuda.current_device())

# load already prepared data

train1 = pickle.load(open('Data/pdata_3hosp/h143_train', 'rb'), encoding='bytes')
valid1 = pickle.load(open('Data/pdata_3hosp/h143_valid', 'rb'), encoding='bytes')
test1 =  pickle.load(open('Data/pdata_3hosp/h143_test', 'rb'), encoding='bytes')



def model_run(epochs, ehr_model,optimizer,batch_size,w_model,fnme):
 
  bestValidAuc = 0.0
  bestTestAuc = 0.0
  bestValidEpoch = 0
  
  for ep in range(epochs):
      current_loss, train_loss = TVT.train(train1, model= ehr_model, optimizer = optimizer, batch_size = batch_size)
      avg_loss = np.mean(train_loss)
      valid_auc, y_real, y_hat  = TVT.calculate_auc(model = ehr_model, data = valid1, which_model = w_model, batch_size = batch_size)
      if valid_auc > bestValidAuc: 
          bestValidAuc = valid_auc
          bestValidEpoch = ep
          best_model= ehr_model
          #shortTestAUC, y_real, y_hat  = TVT.calculate_auc(model = ehr_model, data = test_sh_L, which_model = w_model, batch_size = batch_size)
          #longTestAUC, y_real, y_hat  = TVT.calculate_auc(model = ehr_model, data = test_l_L, which_model = w_model, batch_size = batch_size)

      if ep - bestValidEpoch >12:
          break
          
  bmodel_pth='models/'+fnme
  bestTestAuc, y_real, y_hat = TVT.calculate_auc(model = best_model, data = test1, which_model = w_model, batch_size = batch_size)
  torch.save(best_model, bmodel_pth)
  buf = '|%f |%f |%d ' % (bestValidAuc, bestTestAuc, bestValidEpoch )
  #buf = '|%f |%f |%d |%f |%f' % (bestValidAuc, bestTestAuc, bestValidEpoch , shortTestAUC ,longTestAUC)
  


  return bestValidAuc , buf
