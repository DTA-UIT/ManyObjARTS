import os, sys
import random
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NASBench.NASBench import NASBench
from nats_bench import create
from pprint import pprint
import torch
from ZeroCostNas.foresight.models.nasbench2 import get_model_from_arch_str
from ZeroCostNas.foresight.pruners import predictive
from ZeroCostNas.foresight.weight_initializers import init_net

def get_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10 if args.dataset == 'cifar10' else 120


class NATS(NASBench):
    def __init__(self):
        super().__init__()
        self.op_names = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]

        """
        Call API
        """
        url = os.path.dirname(__file__)
        self.api = create(f"{url[:-len('/NASBench')] + '/source/NATS-tss-v1_0-3ffb9-simple/'}", 'tss', fast_mode=True, verbose=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def convert_individual_to_query(self, ind):
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
    
    def query_bench(self, ind, dataset, epoch, measure):
        """
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
        """
        self.convert_individual_to_query(ind)
        arch_index = self.api.query_index_by_arch(self.cell)
        if self.api.query_index_by_arch(self.cell) == -1:
            raise Exception('Invalid NATSBench cell')
        query_bench = self.api.get_more_info(arch_index, dataset, hp=epoch, is_random=False)
        
        return query_bench if measure == None else query_bench[measure] 
    
    def evaluate_arch(self, args, ind, dataset, measure, train_loader, use_csv=False, epoch=None):
        """
        Hàm đánh giá kiến trúc bằng cách truy xuất NATS Bench.
        
        Arguments:
        args -- argparse truyền vào
        ind -- Cá thể cần được đánh giá.
        dataset -- Dataset để đánh giá kiến trúc (cifar10, cifar100, Imagenet16-120).
        measure -- Phương pháp đánh giá (test-accuracy, synflow,...).
        train_loader 
        use_csv -- Có sử dụng truy vấn thông tin từ file log
        epoch -- Real training tại epoch thứ mấy trong bench (nếu sử dụng accuracy)

        Returns:
        value -- Accuracy của cá thể ind với measure được sử dụng tại dataset đang xét (cifar 10, cifar 100, Imagenet16-120).
        """
        
        
        proxy_log = {
            'synflow': synflow_log,
            'jacob_cov': jacov_log,
            'test-accuracy': testacc_log,
            'flops': flops_log,
            'valid-accuracy': None,
            'train-accuracy': None
        }

        result = {
            'flops': 0,
            'params': 0,
            'test-accuracy': 0,
            'train-accuracy': 0,
            'valid-accuracy': 0,
            'synflow': 0,
            'jacob_cov': 0,
            'snip': 0,
            'grasp': 0,
            'fisher': 0
        }

        if use_csv:

            def get_index_csv (ind):
                index = 0
                for i in range(len(ind)):
                    index += ind[i] * pow(5, len(ind) - i - 1)
                return index

            arch_index = get_index_csv (ind)

            """ 
            Result -> synflow, jacob_cov, test-acc, flops
            """
            result[measure] = proxy_log[measure][arch_index]

            if measure == 'jacob_cov' and np.isnan(result['jacob_cov']):
                result['jacob_cov'] = -1e9

        else:
            if epoch is not None and measure in ['test-accuracy', 'train-accuracy', 'valid-accuracy']: # Return from NASBench
                result[measure] = self.query_bench(ind, dataset, epoch, measure)

            elif measure == 'flops':
                self.convert_individual_to_query(ind)
                arch_index = self.api.query_index_by_arch(self.cell)
                info = self.api.get_cost_info(arch_index, dataset)
                result[measure] = info[measure]
                
            else:
                
                cell = get_model_from_arch_str(arch_str=self.convert_individual_to_query(ind), num_classes=get_num_classes(args))
                net = cell.to(self.device)
                init_net(net, args.init_w_type, args.init_b_type)

                torch.manual_seed(args.seed) 
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                measures = predictive.find_measures(net, 
                                                    train_loader, 
                                                    (args.dataload, args.dataload_info, get_num_classes(args)), 
                                                    args.device, measure_names=measure)    
                
                result[measure] = measures if not np.isnan(measures) else -1e9
            
        return result[measure]