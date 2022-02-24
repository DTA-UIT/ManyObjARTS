import time
import numpy as np
from algorithm.pymoo.pymoo.core.problem import Problem
from algorithm.utils.algorithm import Algorithm
from algorithm.pymoo.pymoo.factory import get_performance_indicator
from search201 import NATSBench

class NASBench101(NATSBench):
    def __init__(self, n_var, n_obj, xl, xu, api, dataset, pareto_front_url, proxy_log):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=xl, xu=xu,
                         api=api, dataset=dataset, pareto_front_url=pareto_front_url, proxy_log=proxy_log)
    
    def calc_IGD(self, pop, objectives):
        return super().calc_IGD(pop, objectives)
    
    def _evaluate(self, designs, out, *args, **kwargs):
        super()._evaluate(designs, out, *args, **kwargs)
        