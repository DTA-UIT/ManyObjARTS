import os, sys 
url = os.path.dirname(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from source.NASBenchHW.hw_nas_bench_api import HWNASBenchAPI as HWAPI
import random
import numpy as np
import matplotlib.pyplot as plt

hw_api = HWAPI(f"{url[:-len('NASBench')] + 'source/NASBenchHW/HW-NAS-Bench-v1_0.pickle'}", search_space="fbnet")


def query_bench(ind, dataset, metric=None):
    """
    Arguments:
    ind -- individual to query
    dataset -- dataset to query ('cifar100', 'ImageNet')
    metric (optional) -- metric to query ('edgegpu_latency', 
                                          'edgegpu_energy', 
                                          'raspi4_latency', 
                                          'pixel3_latency', 
                                          'eyeriss_latency', 
                                          'eyeriss_energy', 
                                          'fpga_latency', 
                                          'fpga_energy', 
                                          'average_hw_metric')
    """
    if not isinstance(ind, list): # If ind is not list then transform it into list
        ind = ind.tolist()
    query = hw_api.query_by_index(ind, dataset)
    return query if metric is None else query[metric]