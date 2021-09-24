import numpy as np 
import pandas as pd 
import os

import warnings 
warnings.filterwarnings("ignore") 
import time 
import random
import yaml 

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
