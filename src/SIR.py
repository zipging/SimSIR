import torch
import torch.nn as nn
import SpaGCN as spg
import pickle
import scanpy as sc
from SIR.src.main import driver
from SIR.src.main.clustering.DEC import DEC
from SIR.src.main._config import Config
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
    # please choose the 151676 for testing
    def get_data(sample_id, data_type='image'):
        assert sample_id == '151676', "please choose the 151676 for testing"
        if data_type == 'image':
            path_file = f"SIR/data/DLPFC_matrix_{sample_id}.dat"
            with open(path_file,'rb') as f:
                all_gene_exp_matrices = pickle.load(f)
            all_gmat = {k:all_gene_exp_matrices[k].todense() for k in list(all_gene_exp_matrices.keys())}
            dataset=np.array(list(all_gmat.values()))
            
            return dataset
        else:
            path_file = f"SIR/data/{sample_id}_10xvisium.h5ad"
            adata = sc.read_h5ad(path_file)
        
            return adata
    
    def data_process(adata):
        adata = adata[~adata.obs['layer_guess_reordered_short'].isna()]
        adata = adata[~adata.obs['discard'].astype(bool), :]
        adata.obs['cluster']  = adata.obs['layer_guess_reordered_short'] 
        adata.var_names_make_unique()
        spg.prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
        spg.prefilter_specialgenes(adata)
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        all_genes = adata.var.index.values
        adata.obs['array_x']=np.ceil((adata.obs['array_col']-
                                  adata.obs['array_col'].min())/2).astype(int)
        adata.obs['array_y']=adata.obs['array_row']-adata.obs['array_row'].min()
        all_gene_exp_matrices = {}
        shape = (adata.obs['array_y'].max()+1, adata.obs['array_x'].max()+1)
        
        for gene in tqdm(all_genes, desc="adata2image", unit="gene"):
            g_matrix = np.zeros(shape=shape)
            g = adata[:,gene].X.todense().tolist()
            c = adata.obsm['spatial']
            for i,row_col in enumerate(zip(adata.obs['array_y'],adata.obs['array_x'])):
                
                row_ix,col_ix = row_col
                # print(f'{row_ix},{col_ix}')
                g_matrix[row_ix,col_ix] = g[i][0]
            all_gene_exp_matrices[gene] = csr_matrix(g_matrix)
        all_gmat = {k:all_gene_exp_matrices[k].todense() for k in list(all_gene_exp_matrices.keys())}
        dataset=np.array(list(all_gmat.values()))
        
        return dataset
    
    def train(train_loader, pretrain = True):
        image_size = (32, 32)
        config = Config(dataset='cifar', model='MAE').get_parameters()
        cuda = torch.cuda.is_available()
        print("use cuda: {}".format(cuda))
        device = torch.device("cuda:0" if cuda else "cpu")
        model = driver.model('Gene image', 'MAE', config, image_size)
        model.to(device)
        if config['decoder'] == 'Gene image':
            model.pretrain(train_loader=train_loader, 
                           batch_size=config['batch_size'], 
                           lr=config['lr'],
                           pretrain = pretrain)
        y_pred, z, model= DEC(model, train_loader, config = config)
        #return model
        return y_pred, z, model
        #return model

