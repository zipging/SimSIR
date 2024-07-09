import argparse
import torch
import math
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from torch.nn import Linear
from torchlars import LARS
from sklearn.preprocessing import StandardScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler



def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def get_embedding(model, train_loader):
    z_part = []
    q_part = []
    with torch.no_grad():
        for (q, _), (k, _) in train_loader:
            q = q.cuda()
            _, batch_result_z, _, _, batch_result_q = model(q)
            z_part.append(batch_result_z)
            q_part.append(batch_result_q)
    # Concatenate the results along the batch dimension
    z = torch.cat(z_part, dim=0)
    q = torch.cat(q_part, dim=0)
    
    return z, q

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = (np.array(linear_assignment(w.max() - w))).transpose()
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def DEC(model, train_loader, config):
    
    batch_size = config['batch_size_dec']
    
    ####optimizer zetting####
    epochs=100
    base_lr = 4.8
    final_lr = 0
    wd = 1e-6
    warm_epochs = 10
    start_warm = 0
    awl = AutomaticWeightedLoss(3)
    optimizer = torch.optim.SGD([
    {'params': model.parameters()},
    {'params': awl.parameters()}
], lr=4.8, momentum=0.9, weight_decay=1e-6)
    optimizer = LARS(optimizer=optimizer, trust_coef=0.001)
    warmup_lr_schedule = np.linspace(start_warm, base_lr, len(train_loader.loader1.dataset) * warm_epochs)
    iters = np.arange(len(train_loader.loader1.dataset) * (epochs - warm_epochs))
    cosine_lr_schedule = np.array([0 + 0.5 * (base_lr - final_lr) * (1 + \
                         math.cos(math.pi * t / (len(train_loader.loader1.dataset) * (epochs - warm_epochs)))) for t in iters])
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    
    ###get initial embedding###
    z, q = get_embedding(model, train_loader)

    ###initial cluster assignment###
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(z.data.cpu().numpy())
    kmeans = KMeans(n_clusters=config['n_clusters'], init='k-means++', n_init=10)
    y_pred = kmeans.fit_predict(X_scaled)
    y_pred = np.array(y_pred)
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    
    ###finetune model with dec###
    model.train()
    for epoch in tqdm(range(30), desc="Clustering"):
        z, q = get_embedding(model, train_loader)
            
        # update target distribution p
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(z.data.cpu().numpy())
        gmm = GaussianMixture(n_components=config['n_clusters'], random_state=42)
        gmm.fit(X)

        # obtain posterior probability
        tem_q = gmm.predict_proba(X)
        tem_q = torch.tensor(tem_q).to('cuda')
        p = target_distribution(tmp_q)
            
        # evaluate clustering performance
        y_pred = tmp_q.cpu().numpy().argmax(1)
        delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
        y_pred_last = y_pred
        y_pred_count = torch.tensor(y_pred_last)
        counts = torch.bincount(y_pred_count)

        if epoch > 0 and delta_label < config['tol']:
            print('delta_label {:.4f}'.format(delta_label), '< tol',
                      config['tol'])
            print('Reached tolerance threshold. Stopping training.')
            break
        
        ###clustering process###
        it = 0
        total_loss = 0.
        total_kl_loss = 0.
        total_reconstr_loss = 0.
        for (data_q, _), (_, _), idx in train_loader:
            data_q = data_q.to(device)
            iteration = epoch * len(train_loader.loader1.dataset) + it
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[iteration]

            x_bar, _, reconstr_loss, _, q = model(data_q)
            kl_loss = F.kl_div(q.log(), p[idx])
            loss = awl(kl_loss, reconstr_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss = total_loss + reconstr_loss.item()+kl_loss.item()
            total_reconstr_loss = total_reconstr_loss + reconstr_loss.item()
            total_kl_loss = total_kl_loss + kl_loss.item()
            it+=1
            
        print(f"Epoch {epoch + 1}\n"
              f"rec_loss: {total_reconstr_loss / len(train_loader.loader1.dataset)}\n"
              f"kl loss: {total_kl_loss / len(train_loader.loader1.dataset)}\n"
              f"total_loss: {total_loss / len(train_loader.loader1.dataset)}")
        
        z, q = get_embedding(model, train_loader)
            
        # update target distribution p
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(z.data.cpu().numpy())

    
    return y_pred_last, X_scaled, model
