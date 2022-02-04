import os, sys
import random
import numpy as np
from NASBench.NASBench import NASBench
from nats_bench import create
from pprint import pprint

class NATS(NASBench):
    def __init__(self, ind):
        super().__init__(ind)
        self.op_names = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

        """
        Call API
        """
        url = os.path.dirname(__file__)
        self.api = create(f"{url[:-len('/NASBench')] + '/source/NATS-tss-v1_0-3ffb9-simple/'}", 'tss', fast_mode=True, verbose=False)

    
    def convert_individual_to_query(self):
        self.cell = ''
        node = 0
        for i in range(len(self.ind)):
            gene = self.ind[i]
            self.cell += '|' + self.op_names[gene] + '~' + str(node)
            node += 1
            if i == 0 or i == 2:
                node = 0
                self.cell += '|+'
        self.cell += '|'
        return self.cell
    
    def query_bench(self, dataset, epoch, metric=None):
        """
        Arguments
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
        self.convert_individual_to_query()
        arch_index = self.api.query_index_by_arch(self.cell)
        if self.api.query_index_by_arch(self.cell) == -1:
            raise Exception('Invalid NATSBench cell')
        query_bench = self.api.get_more_info(arch_index, dataset, hp=epoch, is_random=False)
        return query_bench if metric == None else query_bench[metric] 