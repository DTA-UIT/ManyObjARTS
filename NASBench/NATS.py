import os, sys
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NASBench.NASBench import NASBench
from numpy import genfromtxt
from nats_bench import create
from pprint import pprint
import torch
from ZeroCostNas.foresight.models.nasbench2 import get_model_from_arch_str
from ZeroCostNas.foresight.pruners import predictive
from ZeroCostNas.foresight.weight_initializers import init_net
from ZeroCostNas.OpCounter.thop import profile

def get_num_classes(args):
    if args.dataset == 'cifar100':
        return 100
    if args.dataset == 'cifar10':
        return 10
    if args.dataset == 'imagenet':
        return 120
    raise Exception('Unknown dataset')

class NATS(NASBench):
    def __init__(self, use_colab=True, debug=False):
        super().__init__()
        
        # Define operations
        self.op_names = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
        url = os.path.dirname(__file__)

        # Call API
        if debug:
            self.api = None
        else:
            if use_colab:
                self.api = create("/content/drive/MyDrive/DTA/NATS Bench/NATS-tss-v1_0-3ffb9-simple", 'tss', fast_mode=True, verbose=False)
            else:
                self.api = create(f"{url[:-len('/NASBench')] + '/source/NATS-tss-v1_0-3ffb9-simple/'}", 'tss', fast_mode=True, verbose=False)
            
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Running on device: {self.device}')
    
    def convert_individual_to_query(self, ind):
        """
        Function to convert an individual to a architecture string
        
        Arguments:
        ind -- Individual to convert
        
        Returns:
        cell -- Architecture string
        """

        self.cell = ''
        node = 0
        for i in range(len(ind)):
            gene = ind[i]
            self.cell += '|' + self.op_names[gene] + '~' + str(node)
            node += 1
            if i == 0 or i == 2:
                node = 0
                self.cell += '|+'
        self.cell += '|'
        return self.cell
    
    def query_bench(self, ind, dataset, epoch, measure=None):
        """
        Function to query NASBench API
        
        Arguments
        dataset -- Dataset to query ('cifar10', 'cifar100', 'ImageNet16-120')
        epoch -- Epoch to query (12, 200) 
        measure (optional) -- Metric to query ('train-loss', 
                                            'train-accuracy', 
                                            'train-per-time', 
                                            'train-all-time', 
                                            'test-loss', 
                                            'test-accuracy', 
                                            'test-per-time', 
                                            'test-all-time')
        
        Returns
        query_bench -- query results (with or without a specific measure)
        """
        self.convert_individual_to_query(ind)
        arch_index = self.api.query_index_by_arch(self.cell)
        
        if self.api.query_index_by_arch(self.cell) == -1:
            raise Exception('Invalid NATSBench cell')
            
        query_bench = self.api.get_more_info(arch_index, dataset, hp=epoch, is_random=False)
        
        return query_bench if measure == None else query_bench[measure] 
    
    def evaluate_arch(self, ind, dataset, measure, args=None, train_loader=None, use_csv=False, proxy_log=None, epoch=None) -> float:
        """
        Function to evaluate an architecture
        
        Arguments:
        args -- Argparse to pass through
        ind -- Evaluating individual
        dataset -- Evaluating dataset ('cifar10', 
                                    'cifar100', 
                                    'Imagenet16-120').
        measure -- Evaluation method ('train-loss', 
                                    'train-accuracy', 
                                    'train-per-time', 
                                    'train-all-time', 
                                    'test-loss', 
                                    'test-accuracy', 
                                    'test-per-time', 
                                    'test-all-time', 
                                    'flops',
                                    'macs',
                                    'latency',
                                    'params',
                                    'synflow',
                                    'jacob_cov',
                                    'snip',
                                    'grasp',
                                    'fisher').
        train_loader -- Data train loader
        use_csv (optional) -- To choose whether to use csv file to get results (Bool)
        proxy_log (optional, but required if use_csv is True) -- Log file url [synflow, jacov, test-acc, flops]
        epoch -- If measure is accuracy, this is the epoch to evaluate (int)

        Returns:
        result[measure] -- Result of evaluation at the present dataset
        """
        
        if use_csv and not proxy_log:
            raise Exception('No proxy log to query csv')
        
        if (measure == 'test-accuracy' or measure == 'train-accuracy') and epoch == None:
            raise Exception('No specific epoch for test/train accuracy')
        

        result = {
            'flops': 0,
            'params': 0,
            'latency': 0,
            'macs': 0,
            'test-accuracy': 0,
            'train-accuracy': 0,
            'valid-accuracy': 0,
            'synflow': 0,
            'jacob_cov': 0,
            'snip': 0,
            'grasp': 0,
            'fisher': 0
        }

        # If use log file, then get results from csv file
        if use_csv:
            proxy_file = proxy_log   
            proxy_log = genfromtxt(proxy_log, delimiter=',')     
            
            def get_index_csv (ind):
                index = 0
                for i in range(len(ind)):
                    index += ind[i] * pow(5, len(ind) - i - 1)
                return index

            arch_index = get_index_csv(ind) # Get index of architecture from log file

            result[measure] = proxy_log[arch_index]

            if measure == 'jacob_cov' and np.isnan(result['jacob_cov']):
                result['jacob_cov'] = -1e9

            if 'synflow' in proxy_file: 
                synflow_result = np.mean(result[measure])           
                return synflow_result
            
        else:
            # If measure is accuracy, query at a specific epoch
            if epoch is not None and measure in ['test-accuracy', 'train-accuracy', 'valid-accuracy']: 
                result[measure] = self.query_bench(ind, dataset, epoch, measure)

            # If get flops info, then query from NATSBench
            elif measure in ['flops', 'latency', 'params']:
                self.convert_individual_to_query(ind)
                arch_index = self.api.query_index_by_arch(self.cell)
                info = self.api.get_cost_info(arch_index, dataset)
                result[measure] = info[measure]
            
            elif measure == 'macs':
                if train_loader == None:
                    raise Exception('No train loader to get MACs')
                if args.dataset == 'cifar10' or args.dataset == 'cifar100':    
                    input = torch.randn(len(train_loader), 3, 32, 32)
                elif args.dataset == 'imagenet':
                    input = torch.randn(len(train_loader), 3, 16, 16)
                else:
                    raise Exception('Unsupported dataset')
                result['macs'], _ = profile(self.cell, inputs=(input, ), verbose=False)   
                
            # If None of above, then evaluate the architecture using zero-cost proxies
            else: 
                if args == None and train_loader == None:
                    raise Exception('No argparse or no train loader for Zero-Cost Proxies')

                cell = get_model_from_arch_str(arch_str=self.convert_individual_to_query(ind), num_classes=get_num_classes(args))
                init_net(cell, args.init_w_type, args.init_b_type)
                net = cell.to(self.device)        
                measures = predictive.find_measures(net, 
                                                    train_loader, 
                                                    (args.dataload, args.dataload_info, get_num_classes(args)), 
                                                    self.device,
                                                    measure_names=[measure])    
                
                result[measure] = measures[measure] if not np.isnan(measures[measure]) else -1e9
            
        return result[measure]
    
    def evaluate_multi_measure(self, args, ind, dataset, measure, train_loader, use_csv=False, proxy_log=None, epoch=200) -> dict:
        result = {}
        for seperated_measure in measure:
            result[seperated_measure] = self.evaluate_arch(args, ind, dataset, seperated_measure, train_loader, use_csv, proxy_log, epoch)
        return result