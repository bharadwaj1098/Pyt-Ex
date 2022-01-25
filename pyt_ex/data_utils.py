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


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    medium article :- https://towardsdatascience.com/better-data-loading-20x-pytorch-speed-up-for-tabular-data-e264b9e34352
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
