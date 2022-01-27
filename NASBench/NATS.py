import os, sys
import random
import numpy as np
from nats_bench import create
from pprint import pprint
import matplotlib.pyplot as plt

"""
Call API
"""
url = os.path.dirname(__file__)
api = create(f"{url[:-len('/NASBench')] + '/source/NATS-tss-v1_0-3ffb9-simple/'}", 'tss', fast_mode=True, verbose=False)

def convert_individual_to_str( ind ):
    st = ''
    op_names=["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
    node = 0
    for i in range(len(ind)):
        gene = ind[i]
        st += '|' + op_names[gene] + '~' + str(node)
        node += 1
        if i == 0 or i == 2:
            node = 0
            st += '|+'
    st += '|'
    return st

def query_bench(ind, dataset, epoch, metric=None):
    """
    Arguments
    ind -- Individual to query (ex: [3, 4, 2, 4, 4, 1])
    dataset -- Dataset to query ('cifar10', 'cifar100', 'ImageNet16-120')
    epoch -- Epoch to query (12, 200)
    metric (optional) -- Metric to query ('train-loss', 
                                          'train-accuracy', 
                                          'train-per-time', 
                                          'train-all-time', 
                                          'test-loss', 
                                          'test-accuracy', 
                                          'test-per-time', 
                                          'test-all-time')
    """
    arch_str = convert_individual_to_str(ind)
    arch_index = api.query_index_by_arch(arch_str)
    if api.query_index_by_arch(arch_str) == -1:
        raise Exception('Invalid NATSBench cell')
    query_bench = api.get_more_info(arch_index, dataset, hp=epoch, is_random=False)
    return query_bench if metric == None else query_bench[metric]