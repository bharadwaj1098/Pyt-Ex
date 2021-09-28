import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm

import warnings 
warnings.filterwarnings("ignore") 
import time 
import random
import yaml 

import torch 
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torchvision.transforms as transforms
from torch._C import device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  


class dynamic_nn(nn.Module):
    '''
    TODO :- 
        1) Batch_norm layers
        2) Graphics
        3) deeper layers
        4) Weighted sampling for class imbalance

    Important points:-
        1) No need to add the log_softmax activation to the last layer 
        as cross_entropy loss does that for you while training.
        2) However log_softmax need to be applied while validation and testing 
        i.e; on the model output predictions and the next step would be the arg_max
    '''
    def __init__(self, dic):
        super(dynamic_nn, self).__init__()
        self.dic=dic
        self.struct = dic['input_network']
        self.activations = [i for i in dic['activation']]
        self.criterion = getattr(nn, dic['loss_fn'])()
        self.optimizer = getattr(optim, dic['optimizer'])()

        if len(self.struct) == 3:
            self.model = nn.Sequential(
                            nn.Linear(self.struct[0], self.struct[1]),
                            getattr(nn, self.activations[0])(),
                            nn.Linear(self.struct[1], self.struct[2])
                        )
        elif len(self.struct) == 4:
            self.model = nn.Sequential(
                            nn.Linear(self.struct[0], self.struct[1]),
                            getattr(nn, self.activations[0])(),
                            nn.Linear(self.struct[1], self.struct[2]),
                            getattr(nn, self.activations[1])(),
                            nn.Linear(self.struct[2], self.struct[3])
                        )
    
    def forward(self, x):
        return self.model(x)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def show(self):
        print(self.model)

    def sample_data(self, size):
        '''
        By using TensorDataset you need not use 
        '''
        X = torch.rand(size,10)
        y = torch.rand(size, 1)
        X, y = X.to(device), y.to(device)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size = size, shuffle=True)

    def train_val(self, train_dataloader, val_dataloader=None):
        '''
        We’re using the nn.CrossEntropyLoss because this is a multiclass classification problem. 
        We don’t have to manually apply a log_softmax layer after our final layer because nn.CrossEntropyLoss does that for us. 
        However, we need to apply log_softmax for our validation and testing.
        
        default validation datloader is none but, if given a dataloader then the model will use it.

        add loss and accuracy of each minibatch to average it for loss of whole epoch and accuracy
        '''
        print("Begin_training")
        optimizer = self.optimizer(self.model.parameters(), lr = self.dic['learning_rate'])
        self.train_loss_list, self.val_loss_list = [], []

        for i in tqdm(range(1, self.dic['epochs']+1 )):
            train_epoch_loss, val_epoch_loss = 0, 0
            train_epoch_acc, val_epoch_acc = 0, 0
            self.model.train()
            for X, y in train_dataloader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()

                y_pred = self.model(X)
                
                train_loss = self.criterion(y_pred, y)
                train_acc = self.multi_acc(y_pred, y)

                train_loss.backward()
                optimizer.step()

                train_epoch_loss += train_loss.item() 
                train_epoch_acc += train_acc

            
            with torch.no_grad():
                self.model.eval()
                for X, y in val_dataloader:
                    X, y = X.to(device), y.to(device)
                    y_pred = self.model(X)
                    val_loss = self.criterion(y_pred, y)
                    val_acc = self.multi_acc(y_pred, y)

                    val_epoch_loss += val_loss
                    val_epoch_acc += val_acc 


    def multi_acc(self, y_pred, y_test):

        if self.criterion=="CrossEntropyLoss":
            y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
        
        elif self.criterion=="MSELoss":
            _, y_pred_tags = torch.max(y_pred, dim = 1)    
            
        correct_pred = (y_pred_tags == y_test).float()
        accuracy = correct_pred.sum() / len(correct_pred)
        accuracy = torch.round(accuracy * 100)
        return accuracy
        



