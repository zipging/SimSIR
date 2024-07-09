import torch
import torch.nn as nn
import pickle
from SIR.src import driver
from SIR.src.clustering.DEC import DEC
from SIR.src._config import Config
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix
import numpy as np
import json
import os

'''
What you should input
dataset: gene image data with size [N, 1, 72, 59]
total: gene similarity matrix with size [N, N]

What you will get return
model: encoder that haved been trained
y_pred: label that SIR generative
embedding: embedding that generatived by encoder

Others
If you wanna get other return such as x_bar or parameters of SMM, just rewrite DEC to get what you want.
'''

class SIR():
    def train(train_loader, pretrain = True):
        image_size = (32, 32)
        config = Config(dataset='cifar', model='MAE').get_parameters()
        cuda = torch.cuda.is_available()
        print("use cuda: {}".format(cuda))
        device = torch.device("cuda:0" if cuda else "cpu")
        model = driver.model('Gene image', 'MAE', config, image_size)
#         if torch.cuda.device_count() > 1:
#             print(f"Let's use {torch.cuda.device_count()} GPUs!")
    
#         model.cuda()  # 确保模型已移至 GPU
#         device_ids = [0, 1]
#         model = nn.DataParallel(model, device_ids=device_ids)
        if config['decoder'] == 'Gene image':
            model.pretrain(train_loader=train_loader, 
                           batch_size=config['batch_size'], 
                           lr=config['lr'],
                           pretrain = pretrain)
        y_pred, z, model= DEC(model, train_loader, config = config)
        #return model
        return z, model
        #return model

