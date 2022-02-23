import os, sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pymoo.pymoo.core.problem import Problem

class ProblemWrapper(Problem):
    def __init__(self):
        super().__init__()
        self.generation_count = 0
    
    def _evaluate(self, x, out, *args, **kwargs):
        start = time.time()
        print(f'Gen: {generation_count} - seed: {seed}')


        flops, testacc, synflow = [], [], []

        for design in designs:
            flops.append(api.evaluate_arch(args=argparse, ind=design, dataset=dataset, measure='flops', use_csv=True, train_loader=train_loader, proxy_log=proxy_log))
            testacc.append(api.evaluate_arch(args=argparse, ind=design, dataset=dataset, measure='test-accuracy', epoch=200, use_csv=True, train_loader=train_loader, proxy_log=proxy_log))
            synflow.append(-1 * api.evaluate_arch(args=argparse, ind=design, dataset=dataset, measure='synflow', use_csv=True, train_loader=train_loader, proxy_log=proxy_log))

        objectives = np.stack((flops, synflow), axis=-1)
        testacc_flops = np.array(np.stack((flops, testacc), axis=-1))
        calc_IGD(pop=designs, generation_count=generation_count, seed=seed, objectives=objectives)

        out['F'] = np.array(objectives)
        end = time.time()
        elapsed_time = end - start
        print('time:', elapsed_time)

        res_of_run['time'].append(elapsed_time)
        res_of_run['log_testacc'].append(testacc)
        res_of_run['log_objectives'].append(objectives)
        res_of_run['log_pop'].append(designs)

        generation_count +=1    