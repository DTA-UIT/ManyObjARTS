"""
NATS-Bench
"""
# from numpy import genfromtxt
# import argparse
# import NAS
# from NAS.ZeroCostNas.foresight.dataset import get_cifar_dataloaders

# dataset = 'cifar10'

# train_loader, test_loader = get_cifar_dataloaders(64, 64, dataset, 2)
# dataset_args = 'imagenet' if dataset == 'ImageNet16-120' else dataset

# args = argparse.Namespace(api_loc='', 
#                           outdir='',
#                           init_w_type='none', # Initialize weight
#                           init_b_type='none', # Initialize bias
#                           batch_size=64,      # Batch size
#                           dataset=dataset_args,
#                           gpu=0,
#                           num_data_workers=2,
#                           dataload='random',
#                           dataload_info=1,
#                           seed=1,
#                           write_freq=1,
#                           start=0,
#                           end=0,
#                           noacc=False
#                           )

# from NAS.NASBench import NATS

# api = NATS.NATS(use_colab=False)
# api.evaluate_arch(args=args, ind=[2, 3, 1, 0, 4, 2], dataset=dataset, measure='synflow', train_loader=train_loader)


"""
NAS-Bench-101
"""
from NASBench import NAS101
from ZeroCostNas.foresight.dataset import get_cifar_dataloaders
import numpy as np
import argparse

dataset = 'cifar10'

train_loader, test_loader = get_cifar_dataloaders(64, 64, dataset, 2)
dataset_args = 'imagenet' if dataset == 'ImageNet16-120' else dataset

args = argparse.Namespace(api_loc='', 
                          outdir='',
                          init_w_type='none', # Initialize weight
                          init_b_type='none', # Initialize bias
                          batch_size=64,      # Batch size
                          dataset=dataset_args,
                          gpu=0,
                          num_data_workers=2,
                          dataload='random',
                          dataload_info=1,
                          seed=1,
                          write_freq=1,
                          start=0,
                          end=0,
                          noacc=False
                          )

api = NAS101.NAS101()
ind = [2, 3, 0, 2, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0,
        1, 1, 1]

def convert_ind_tri(ind, ops_none=None):
    res = np.zeros((7, 7), dtype=int)
    # print(ind)
    k = 0
    for i in range(7):
        for j in range(i + 1, 7):
            res[i][j] = ind[k]
            k += 1
    if ops_none != None:
        res = np.delete(res, ops_none, axis=1)
        res = np.delete(res, ops_none, axis=0)
    return res 

def get_operations(ind):
    ops_none = []
    ops = ['input']
    for i in range(0, 5):
        if ind[i] == 0:
            ops_none.append(i)
        elif ind[i] == 1:
            ops.append('conv1x1-bn-relu')
        elif ind[i] == 2:
            ops.append('conv3x3-bn-relu')
        else:
            ops.append('maxpool3x3')
    ops.append('output')
    
print(api.evaluate_arch(args, ind=ind[:5], measure='test_accuracy', train_loader=train_loader, epoch=108))