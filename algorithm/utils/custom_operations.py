import os, sys 
import numpy as np

from source.nasbench import nasbench
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pymoo.pymoo.operators.crossover.pntx import PointCrossover
from pymoo.pymoo.operators.crossover.util import crossover_mask
from pymoo.pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem

from NASBench.NAS101 import NAS101 

nasbench101_api = NAS101(debug=True)

class TwoPointsCrossover(PointCrossover):
    def __init__(self):
        super().__init__(n_points = 2)
    
    def _do(self, _, X):
        # get the X of parents and count the matings
        _, n_matings, n_var = X.shape

        # start point of crossover
        r = np.row_stack([np.random.permutation(n_var - 1) + 1 for _ in range(n_matings)])[:, : self.n_points]
        r.sort(axis=1)
        r = np.column_stack([r, np.full(n_matings, n_var)])

        # the mask do to the crossover
        M = np.full((n_matings, n_var), False)

        # original_parent1, original_parent2 = np.copy(X), np.copy(M)

        # create for each individual the crossover range
        for i in range(n_matings):
            j = 0
            
            while j < r.shape[1] - 1:
                a, b = r[i, j], r[i, j + 1]
                maximum_crossover_attempts = len(X)
                print(f"Crossover phase:\n{a}\n{b}\n")
                current_attempt = 0 
                while not nasbench101_api.is_valid(a) or not nasbench101_api.is_valid(b):
                    if current_attempt == maximum_crossover_attempts:
                        break
                else:
                    self._do(a, b)
                    current_attempt += 1 
                
                M[i, a:b] = True
                j += 2    
                
        _X = crossover_mask(X, M)
        
        # if nasbench101_api.is_valid(X) and nasbench101_api.is_valid(M):
        #     _X = crossover_mask(X, M)
        
        # 
        # current_attempt = 0
        # while not nasbench101_api.is_valid(X) or not nasbench101_api.is_valid(M):
        #     if current_attempt == maximum_crossover_attempts:
        #         return crossover_mask(original_parent1, original_parent2)
        #     else:
        #         self._do(M, X)
        #         current_attempt += 1

        return _X
    
class CustomPolynomialMutation(PolynomialMutation):
    def __init__(self, eta, prob=0.0):
        super().__init(eta=eta, prob=prob)
        
    def _do(self, problem, X):
        X = X.astype(float)
        Y = np.full(X.shape, np.inf)

        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        do_mutation = np.random.random(X.shape) < self.prob

        Y[:, :] = X

        xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)[do_mutation]
        xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)[do_mutation]

        X = X[do_mutation]

        delta1 = (X - xl) / (xu - xl)
        delta2 = (xu - X) / (xu - xl)

        mut_pow = 1.0 / (self.eta + 1.0)

        rand = np.random.random(X.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        deltaq = np.zeros(X.shape)

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]

        # mutated values
        _Y = X + deltaq * (xu - xl)

        # back in bounds if necessary (floating point issues)
        _Y[_Y < xl] = xl[_Y < xl]
        _Y[_Y > xu] = xu[_Y > xu]

        # set the values for output
        Y[do_mutation] = _Y

        # in case out of bounds repair (very unlikely)
        Y = set_to_bounds_if_outside_by_problem(problem, Y)

        return Y