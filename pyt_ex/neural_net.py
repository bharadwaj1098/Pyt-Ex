import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm
from matplotlib import pyplot as plt

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


class Ann(nn.Module):
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
    def __init__(self, dic, **kwargs):
        super(Ann, self).__init__()
        # super().__init__(**kwargs)
        self.dic=dic
        self.struct = dic['input_network']
        self.activations = [i for i in dic['activation']]
        #self.criterion = dic['loss_fn'] #getattr(nn, dic['loss_fn'])()
        # self.optimizer = optim.getattr(optim, dic['optimizer'])(self.model.parameters(), lr = self.dic['learning_rate'])

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
        optimizer = getattr(optim, self.dic['optimizer'])(self.model.parameters(), lr = self.dic['learning_rate'])
        criterion = getattr(nn, self.dic['loss_fn'])()
        self.train_loss_list, self.val_loss_list = [], []
        self.train_acc_list, self.val_acc_list = [], []

        for i in tqdm(range(1, self.dic['epochs']+1 )):
            train_epoch_loss, val_epoch_loss = 0, 0
            train_epoch_acc, val_epoch_acc = 0, 0
            self.model.to(device)
            for X, y in train_dataloader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()

                y_pred = self.model(X)

                if self.dic['loss_fn'] == 'MSELoss':
                    _, y_pred = torch.max(y_pred, dim = 1) 
                
                train_loss = criterion(y_pred, y)
                
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
                    val_loss = criterion(y_pred, y)
                    val_acc = self.multi_acc(y_pred, y)

                    val_epoch_loss += val_loss
                    val_epoch_acc += val_acc 
            
            self.train_loss_list.append(train_epoch_loss/len(train_dataloader))
            self.val_loss_list.append(val_epoch_loss/len(val_dataloader))
            self.train_acc_list.append(train_epoch_acc/len(train_dataloader))
            self.val_acc_list.append(val_epoch_acc/len(val_dataloader))

    def multi_acc(self, y_pred, y_test):

        if self.dic['loss_fn']=="CrossEntropyLoss":
            y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
        
        elif self.dic['loss_fn']=="MSELoss":
            _, y_pred_tags = torch.max(y_pred, dim = 1)    
            
        correct_pred = (y_pred_tags == y_test).float()
        accuracy = correct_pred.sum() / len(correct_pred)
        accuracy = torch.round(accuracy * 100)
        return accuracy

    def accuracy_plot(self):
        plt.plot(self.train_acc_list, label = 'train_acc_list')
        plt.plot(self.val_acc_list, label = 'val_acc_list')
        plt.legend()

    def loss_plot(self):
        plt.plot(self.train_loss_list, label = 'train_loss_list')
        plt.plot(self.val_loss_list, label = 'val_loss_list')
        plt.legend()

# class grahics(graphics_parent):
#     def __init__(self):
#         super().__init__()
    
#     def loss_plot(self):
#         plt.plot(self.train_loss_list, label = 'train_loss_list')
#         plt.plot(self.val_loss_list, label = 'val_loss_list')
#         plt.legend()
    
#     def accuracy_plot(self):
#         plt.plot(self.train_acc_list, label = 'train_acc_list')
#         plt.plot(self.val_acc_list, label = 'val_acc_list')
#         plt.legend() 