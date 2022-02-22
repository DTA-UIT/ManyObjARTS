import os, sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
from NASBench.NASBench import NASBench
from ZeroCostNas.foresight.models import nasbench1
from ZeroCostNas.foresight.pruners import predictive
from ZeroCostNas.foresight.weight_initializers import init_net
from source.nasbench.nasbench import api
from thop import profile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_num_classes(args):
    if args.dataset == 'cifar100':
        return 100
    if args.dataset == 'cifar10':
        return 10
    if args.dataset == 'imagenet':
        return 120
    raise Exception('Unknown dataset')

class NAS101(NASBench):
    __model_path = None
    def __init__(self):
        super().__init__()
        
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
        
        url = os.path.dirname(__file__)
        self.api = api.NASBench(f"{url[:-len('/NASBench')] + '/source/nasbench/nasbench_full.tfrecord'}")
        self.cell = None
    
    def query_bench(self, ind, ops, metric=None, epochs=108):
        """
        Arguments:
        metric (optional) --  metric to query ('module_adjacency', 
                                               'module_operations', 
                                               'trainable_parameters', 
                                               'training_time', 
                                               'train_accuracy', 
                                               'validation_accuracy', 
                                               'test_accuracy')
        """  
        self.cell = api.ModelSpec(ind, ops)

        try:
            self.query_result = self.api.query(self.cell) if 'accuracy' not in metric else self.api.query(self.cell, epochs=epochs)
        except:
            print(f"Cell {self.cell.__dict__['original_matrix']} is invalid for NASBench101")    
            self.api._check_spec(self.cell)

        # if not self.api.is_valid(self.cell):
        #     print(self.api._check_spec(self.cell))
        #     raise Exception("Invalid NASBench101 cell") 
        return self.query_result if metric == None else self.query_result[metric] 
    
    
    def is_valid(self, ind, ops):
        self.cell = api.ModelSpec(ind, ops)
        return self.api.is_valid(self.cell)
    
    def evaluate_arch(self, args, ind, ops, measure, train_loader, use_csv=False, proxy_log=None, epoch=None):
        """
        Function to evaluate an architecture
        
        Arguments:
        args -- Argparse to pass through
        ind -- Evaluating individual (DAG representation)
        ops -- Operations list for the individual
        measure -- Evaluation method ('train_accuracy',
                                    'validation_accuracy',
                                    'test_accuracy',
                                    'trainable_parameters',
                                    'training_time',
                                    'flops',
                                    'macs',
                                    'latency',
                                    'synflow',
                                    'jacob_cov',
                                    'snip',
                                    'grasp',
                                    'fisher').
        train_loader -- Data train loader
        use_csv (optional) -- To choose whether to use csv file to get results (Bool)
        proxy_log (optional, but required if use_csv is True) -- Log file [synflow, jacov, test-acc, flops]
        epoch -- If measure is accuracy, this is the epoch to evaluate (4, 12, 36, 108)

        Returns:
        result[measure] -- Result of evaluation at the present dataset
        """
        
        if use_csv and not proxy_log:
            raise Exception("No proxy log to query csv")
        
        if 'accuracy' in measure and epoch == None:
            raise Exception('No epoch to evaluate')
        
        proxy_log = {
            
        }
        
        result = {}
        model = nasbench1.Network(self.cell, 
                                stem_out=128, 
                                num_stacks=3, 
                                num_mods=3,
                                num_classes=get_num_classes(args))

        # If don't use log file, then evaluate directly from NASBench101
        if not use_csv:
            # If measure is 'train_accuracy' or 'validation_accuracy' or 'test_accuracy'
            if epoch != None and 'accuracy' in measure:
                result[measure] = self.query_bench(ind, ops, metric=measure, epochs=epoch)
            
            # If measure is 'trainable_parameters'
            elif measure in ['trainable_parameters']:
                result[measure] = self.query_bench(ind, ops, metric=measure)
            
            # If measure is MACS
            elif measure in ['macs']:
                import torch
                input = torch.randn(1, 3, 32, 32)
                result['macs'], _ = profile(model, inputs=(input, ), verbose=False)
            
            # If use zero-cost methods
            else:
                model.to(args.device)
                init_net(model, args.init_w_type, args.init_b_type)
                
                measures = predictive.find_measures(model, 
                                                    train_loader, 
                                                    (args.dataload, args.dataload_info, get_num_classes(args)), 
                                                    self.device,
                                                    measure_names=[measure])   
            
                result[measure] = measures[measure] if not np.isnan(measures[measure]) else -1e9
            
        # If query from csv file and exists respective log file
        if use_csv and measure in proxy_log:
            pass 
        
        return result[measure]