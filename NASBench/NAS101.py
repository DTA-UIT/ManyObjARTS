import os, sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from source.nasbench.nasbench import api

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

url = os.path.dirname(__file__)

""" 
Load bench 
"""
nasbench = api.NASBench(f"{url[:-len('/NASBench')] + '/source/nasbench/nasbench_full.tfrecord'}")


""" 
Constants for NASBench101
"""
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix

""" 
Create an Inception-like module (5x5 convolution replaced with two 3x3convolutions).
"""
matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
        [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
        [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
        [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],    # max-pool 3x3
        [0, 0, 0, 0, 0, 0, 0]]    # output layer

ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT]


""" 
Query this model from dataset, returns a dictionary containing the metrics associated with this model.
"""
def query_bench(matrix, ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT], metric = None):    
    """
    Arguments:
    matrix -- adjacency matrix of the model
    ops -- operations list
    metric (optional) --  metric to query ('module_adjacency', 
                                           'module_operations', 
                                           'trainable_parameters', 
                                           'training_time', 
                                           'train_accuracy', 
                                           'validation_accuracy', 
                                           'test_accuracy')
    """
    cell = api.ModelSpec(matrix, ops)
    if not nasbench.is_valid(cell):
        raise Exception("Invalid NASBench101 cell") 
    query_bench = nasbench.query(cell)
    return query_bench if metric == None else query_bench[metric]