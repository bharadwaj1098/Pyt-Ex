import numpy as np 
import pandas as pd 
import os

import warnings 
warnings.filterwarnings("ignore") 
import time 
import random
import yaml 
import torch 
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

class preprocessig:
    """
    the current plan is to not handle the categorical variables
    for now add them under drop
    TODO:-
        1) The pre_processing pipeline built before this was bs so re-doing everything.
        2) preprocess the data outside 
    """
    def __init__(self, dic):
        self.dic = dic
        self.raw_df = pd.read_csv(dic['file_name'])
        self.category = dic['category']
        self.numeric = dic['numeric']
        self.drop = dic['drop']
      
    pass 


class ClassifierDataset(Dataset):
    '''
    to set data in a block.
    this dataset will be used by the dataloader to pass the data
    into the model.
    X = float
    y = long
    '''
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)