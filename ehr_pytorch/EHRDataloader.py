
# coding: utf-8

# In[66]:


#general utilities
from __future__ import print_function, division
import os
from os import walk
from tabulate import tabulate
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except:
    import pickle
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode


# In[26]:


#torch libraries 
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
from torchvision import transforms, utils


# In[67]:


# Dataset class loaded from pickles
class EHRdataFromPickles(Dataset):
    def __init__(self, root_dir, file = None, transform=None):
        """
        Args:
            1) root_dir (string): Path to pickled file(s).
                               The directory contains the directory to file(s): specify 'file' 
                               please create separate instances from this object if your data is split into train, validation and test files.               
            2) data should have the format: pickled, 4 layer of lists, a single patient's history should look at this (use .__getitem__(someindex, seeDescription = True))
                [310062,
                 0,
                 [[[0],[7, 364, 8, 30, 10, 240, 20, 212, 209, 5, 167, 153, 15, 3027, 11, 596]],
                  [[66], [590, 596, 153, 8, 30, 11, 10, 240, 20, 175, 190, 15, 7, 5, 183, 62]],
                  [[455],[120, 30, 364, 153, 370, 797, 8, 11, 5, 169, 167, 7, 240, 190, 172, 205, 124, 15]]]]
                 where 310062: patient id, 
                       0: no heart failure
                      [0]: visit time indicator (first one), [7, 364, 8, 30, 10, 240, 20, 212, 209, 5, 167, 153, 15, 3027, 11, 596]: visit codes.
                      
            3)transform (optional): Optional transform to be applied on a sample. Data augmentation related. 
        """
        self.file = None
        if file != None:
            self.file = file
            self.data = pickle.load(open(root_dir + file, 'rb'), encoding='bytes') 
        else:
            print('No file specified')
        self.root_dir = root_dir  
        self.transform = transform 
                                     
    def __getitem__(self, idx, seeDescription = False):
        '''
        Return the patient data of index: idx of a 4-layer list 
        patient_id (pt_sk); 
        label: 0 for no, 1 for yes; 
        visit_time: int indicator of the time elapsed from the previous visit, so first visit_time for each patient is always [0];
        visit_codes: codes for each visit.
        '''
        if self.file != None: 
            sample = self.data[idx]
        else:
            print('No file specified')
        if self.transform:
            sample = self.transform(sample)
        
        vistc = np.asarray(sample[2])
        desc = {'patient_id': sample[0], 'label': sample[1], 'visit_time': vistc[:,0],'visit_codes':vistc[:,1]}     
        if seeDescription: 
            '''
            if this is True:
            You will get a descriptipn of what each part of data stands for
            '''
            print(tabulate([['patient_id', desc['patient_id']], ['label', desc['label']], 
                            ['visit_time', desc['visit_time']], ['visit_codes', desc['visit_codes']]], 
                           headers=['data_description', 'data'], tablefmt='orgtbl'))
        #print('\n Raw sample of index :', str(idx))     
        return sample

    def __len__(self):
        '''
        either return the length of a single file;
        or return the length tuple of train, valid and test file
        '''
        if self.file != None:
            return len(self.data)
        else: 
            print('No file specified')


# In[68]:


#customized parts for dataloader 
def my_collate(batch):
    return list(batch)


# In[94]:


def iter_batch(iterable, samplesize):
    results = []
    iterator = iter(iterable)
    # Fill in the first samplesize elements:
    for _ in range(samplesize):
        results.append(iterator.__next__())
    random.shuffle(results)  # Randomize their positions
    #print(results)
    for i, v in enumerate(iterator, samplesize): #not quite sure what is this for
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items
            #print(v)
    if len(results) < samplesize:
        raise ValueError("Sample larger than population.")
    return results


# In[148]:


def iterbatchloader(loader, batches = 1):
    return iter_batch(loader, batches)


# In[147]:


EHRbatchloader(loader).__next__()


# ### Test to produce either shuffled or non-shuffled batches using our loader

# In[150]:


custom_data_check = EHRdataFromPickles(root_dir = '/data/projects/py_ehr_2/Data/', 
                                       file = 'hf50_cl2_h143_ref_t1.train')
display(custom_data_check.__len__())
display(custom_data_check.__getitem__(5, seeDescription = True))  


# In[113]:


loader = DataLoader(

    custom_data_check,

    batch_size=10,

    shuffle=False,

    collate_fn=my_collate
)


# In[118]:


#iter through batches 
iterbatchloader(loader = loader)


# In[129]:


#or, call next
iterloader = iter(loader)


# In[149]:


iterloader.__next__()

