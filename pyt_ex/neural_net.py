import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import torchbnn as bnn
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
        if self.dic['type'] == 'ann':
            if len(self.struct) == 3:
                self.model = nn.Sequential(
                                nn.Linear(self.struct[0], self.struct[1]),
                                getattr(nn, 'BatchNorm1d')(self.struct[1]),
                                getattr(nn, self.activations[0])(),
                                nn.Linear(self.struct[1], self.struct[2])
                            )
            elif len(self.struct) == 4:
                self.model = nn.Sequential(
                                nn.Linear(self.struct[0], self.struct[1]),
                                getattr(nn, 'BatchNorm1d')(self.struct[1]),
                                getattr(nn, self.activations[0])(),
                                nn.Linear(self.struct[1], self.struct[2]),
                                getattr(nn, 'BatchNorm1d')(self.struct[2]),
                                getattr(nn, self.activations[1])(),
                                nn.Linear(self.struct[2], self.struct[3])
                            )
        
        if self.dic['type'] =='bnn':
            if len(self.struct) == 3:
                self.model = nn.Sequential(
                                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=self.struct[0], out_features=self.struct[1]),
                                getattr(nn, 'BatchNorm1d')(self.struct[1]),
                                getattr(nn, self.activations[0])(),
                                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=self.struct[1], out_features=self.struct[2])
                            )
            elif len(self.struct) == 4:
                self.model = nn.Sequential(
                                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=self.struct[0], out_features=self.struct[1]),
                                getattr(nn, 'BatchNorm1d')(self.struct[1]),
                                getattr(nn, self.activations[0])(),
                                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=self.struct[1], out_features=self.struct[2]),
                                getattr(nn, 'BatchNorm1d')(self.struct[2]),
                                getattr(nn, self.activations[1])(),
                                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=self.struct[2], out_features=self.struct[3]),
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

    def fit(self, train_dataloader, val_dataloader=None, test_dataloader=None, full_dataloader=None):
        '''
        We???re using the nn.CrossEntropyLoss because this is a multiclass classification problem. 
        We don???t have to manually apply a log_softmax layer after our final layer because nn.CrossEntropyLoss does that for us. 
        However, we need to apply log_softmax for our validation and testing.
        
        default validation datloader is none but, if given a dataloader then the model will use it.

        add loss and accuracy of each minibatch to average it for loss of whole epoch and accuracy
        '''
        # print("Begin_training")
        optimizer = getattr(optim, self.dic['optimizer'])(self.model.parameters(), lr = self.dic['learning_rate'])
        criterion = getattr(nn, self.dic['loss_fn'])()
        self.train_loss_list, self.val_loss_list = [], []
        self.train_acc_list, self.val_acc_list = [], []

        for i in range(1, self.dic['epochs']+1 ):#tqdm(range(1, self.dic['epochs']+1 )):
            train_epoch_loss, val_epoch_loss = 0, 0
            # train_epoch_acc, val_epoch_acc = 
            # Y_train = []
            train_prediction = []
            train_ground_truth = []

            self.model.to(device)
            # acc = 0
            for X, y in train_dataloader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()

                y_pred = self.model(X.to(device))
                y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
                train_prediction.append(y_pred_tags.cpu().numpy() ) 
                train_ground_truth.append(y)

                if self.dic['loss_fn'] == 'MSELoss':
                    _, y_pred = torch.max(y_pred, dim = 1) 
                
                train_loss = criterion(y_pred, y)
                train_loss.backward()
                optimizer.step()
                train_epoch_loss += train_loss.item()
            acc = self.multi_acc(self._nested_list(train_prediction), self._nested_list(train_ground_truth))
            
            self.train_acc_list.append(acc)
            self.train_loss_list.append(train_epoch_loss/len(train_dataloader))

            if val_dataloader is not None:
                with torch.no_grad():
                    self.model.eval()
                    val_prediction = []
                    val_ground_truth = []
                    for X, y in val_dataloader:
                        X, y = X.to(device), y.to(device)
                        y_pred = self.model(X)
                        val_loss = criterion(y_pred, y)
                        y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
                        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
                        val_prediction.append(y_pred_tags.cpu().numpy() ) 
                        val_ground_truth.append(y)
                        val_epoch_loss += val_loss.item()
                
                val_acc = self.multi_acc(self._nested_list(train_prediction), self._nested_list(train_ground_truth))
                self.val_acc_list.append(val_acc)
                self.val_loss_list.append(val_epoch_loss/len(val_dataloader))

        with torch.no_grad():
            if test_dataloader is not None:
                self.test_Output, self.test_acc = self.model_final(test_dataloader)
            if full_dataloader is not None:
                self.predictions, self.final_acc = self.model_final(full_dataloader)

    def model_final(self, dataloader=None):
        if dataloader is not None:
            with torch.no_grad():
                prediction = []
                ground_truth = []
                self.model.eval()
                for X_batch, y in dataloader:
                    X_batch = X_batch.to(device)
                    y_test_pred = self.model(X_batch)
                    y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
                    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
                    prediction.append(y_pred_tags.cpu().numpy() ) 
                    ground_truth.append(y)

            output = self._nested_list(prediction)
            y_test = self._nested_list(ground_truth)
            
            return output, self.multi_acc(output, y_test)

    def multi_acc(self, a, b):
        c = 0
        for i in range(len(b)):
            if a[i] == b[i]:
                c+=1
        return c/len(b)

    def _nested_list(self, old_list):
        new_list = []
        for i in old_list:
            for j in i:
                new_list.append(j)
        return new_list