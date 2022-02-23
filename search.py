import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
from algorithm.pymoo.pymoo.core.problem import Problem
from algorithm.utils.algorithm import Algorithm
from algorithm.pymoo.pymoo.factory import get_performance_indicator

class ProblemWrapper(Problem):
    def __init__(self, api, argparse, pareto_front_url, flops_log_url, proxy_log):
        super().__init__()
        self.api = api
        self.archive_phenotype = []
        self.archive_genotype = []
        self.generation_count = 0
        self.seed = 0
        self.dataset = ""
        self.proxy_log = proxy_log # {'test-accuracy': '', 'flops': ''}
        self.argparse = argparse
        self.pareto_front = np.genfromtxt(pareto_front_url, delimiter=',')
        self.flops_log = flops_log_url
        self.pareto_front_normalize = self.pareto_front.copy()
        self.pareto_front_normalize[:, 0] = (self.pareto_front_normalize[:, 0] - self.flops_log.min()) / (self.flops_log.max() - self.flops_log.min())
        self.pareto_front_normalize[:, 1] = self.pareto_front_normalize[:, 1] / 100
        self.res_of_run = dict()
        
    def calc_IGD(self, pop, objectives):
        for i in range(len(objectives)):
            ind_obj = objectives[i]
            archive_phenotype, archive_genotype = Algorithm.get_new_archive(ind_obj, archive_phenotype, archive_genotype, pop[i])
        
            archive_transform_2obj = []
            for ind in archive_genotype:
                performance = self.api.evaluate_arch(ind=ind, dataset=self.dataset, measure='test-accuracy', epoch=200, use_csv=True, proxy_log=self.proxy_log['test-accuracy'])
                flops = self.api.evaluate_arch(ind=ind, dataset=self.dataset, measure='flops', use_csv=True, proxy_log=self.proxy_log['flops'])
                archive_transform_2obj.append((flops, performance))

            archive_transform_2obj = np.array(archive_transform_2obj)
            archive_transform_2obj_normalize = archive_transform_2obj.copy()
            archive_transform_2obj_normalize[:, 0] = (archive_transform_2obj_normalize[:, 0] - self.flops_log.min()) / (self.flops_log.max() - self.flops_log.min())
            archive_transform_2obj_normalize[:, 1] = archive_transform_2obj_normalize[:, 1] / 100

            igd = get_performance_indicator("igd", self.pareto_front)
            igd_normalize = get_performance_indicator("igd", self.pareto_front_normalize)

            self.res_of_run['igd'].append(igd.do(archive_transform_2obj))
            self.res_of_run['igd_normalize'].append(igd_normalize.do(archive_transform_2obj_normalize))
            self.res_of_run['archive_transform_2obj'].append(archive_transform_2obj)
            self.res_of_run['archive_transform_2obj_normalize'].append(archive_transform_2obj_normalize)
            self.res_of_run['archive_genotype'].append(archive_genotype)
            self.res_of_run['archive_phenotype'].append(archive_phenotype)

            print('igd:', igd.do(archive_transform_2obj))
    
    def _evaluate(self, designs, out, *args, **kwargs):
        start = time.time()
        print(f'Gen: {self.generation_count} - seed: {self.seed}')

        flops, testacc, synflow = [], [], []

        for design in designs:
            flops.append(self.api.evaluate_arch(ind=design, dataset=self.dataset, measure='flops', use_csv=True))
            testacc.append(self.api.evaluate_arch(ind=design, dataset=self.dataset, measure='test-accuracy', epoch=200, use_csv=True))
            synflow.append(-1 * self.api.evaluate_arch(ind=design, dataset=self.dataset, measure='synflow', use_csv=True))

        objectives = np.stack((flops, synflow), axis=-1)
        testacc_flops = np.array(np.stack((flops, testacc), axis=-1))
        self.calc_IGD(pop=designs, generation_count=self.generation_count, seed=self.seed, objectives=objectives)

        out['F'] = np.array(objectives)
        end = time.time()
        elapsed_time = end - start
        print('time:', elapsed_time)

        self.res_of_run['time'].append(elapsed_time)
        self.res_of_run['log_testacc_flops'].append(testacc_flops)
        self.res_of_run['log_objectives'].append(objectives)
        self.res_of_run['log_pop'].append(designs)
        self.generation_count +=1    
        