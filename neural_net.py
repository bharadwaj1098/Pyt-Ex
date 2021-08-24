import numpy as np 
#import pandas as pd 
import os 
import yaml 

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
    def __init__(self, dic):
        super(dynamic_nn, self).__init__()

        self.struct = dic['input_network']
        self.activations = [i for i in dic['activation']]
        
        if len(dic['input_network']) - 1 != len(dic['activation']):
            raise ValueError('number of activations are not equal to number of hidden layers')
        
        if self.activations[2] != 'None':
            last_layer_activation = getattr(nn, self.activations[2])()
                                
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
                            nn.Linear(self.struct[2], self.struct[3]), 
                            getattr(nn, self.activations[2])()
                        )
    
    def forward(self, x):
        return self.model(x)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def show(self):
        print(self.model)

    def sample_data(self, size):
        X = torch.rand(size,10)
        y = torch.rand(size, 1)
        X, y = X.to(device), y.to(device)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size = size, shuffle=True)


'''
YET_TODO:-
choice of not adding actvation  for last layer for tasks such as regression

'''

if __name__ == '__main__':
    with open('Input.yaml') as File:
        dic = yaml.load(File, Loader=yaml.FullLoader)

    net = dynamic_nn(dic['FIRST'])
    net = net.to(device)
    
    data = net.sample_data(size = 256)

    optimizer = getattr(optim, dic['FIRST']['optimizer'])(net.parameters(), lr= dic['FIRST']['learning_rate'], weight_decay= dic['FIRST']['weight_decay'])

    loss_fn = getattr(nn, dic['FIRST']['loss'])()
    
    checkpoint = []
    
    LOSS = []

        for i in range(dic['FIRST']['iterations']) 

            X, y = next(iter(data))

            output = net(X)

            loss = loss_fn(output, y).cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
            LOSS.append()

            checkpoint.append(net.state_dict().cpu())
    

        



