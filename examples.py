"""
NASBench 101
"""
from NASBench import NAS101
matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
        [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
        [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
        [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],    # max-pool 3x3
        [0, 0, 0, 0, 0, 0, 0]]    # output layer
query_1 = NAS101.query_bench(matrix) 
query_2 = NAS101.query_bench(matrix, metric='test_accuracy')
print(query_1)
print(query_2)

"""
NATS Benchmark
"""
from NASBench import NATS
ind = [3, 3, 3, 3, 3, 3]
query_1 = NATS.query_bench(ind, 'cifar10', epoch=200)
query_2 = NATS.query_bench(ind, 'cifar10', epoch=200, metric='test-accuracy')
print(query_1)

"""
NASBench 301
"""
from NASBench import NAS301
ind = [6, 3, 4, 6, 2, 4, 4, 6, 0, 0, 2, 1, 2, 0, 3, 0, 1, 3, 3, 6, 3, 6,
        3, 4, 0, 1, 1, 0, 3, 3, 0, 0]

query_1 = NAS301.query_bench(ind)
print(f"Genotype architecture performance: {query_1}")

model_predictor = 'xgb'
query_2 = NAS301.query_bench(ind, model_predictor=model_predictor)
print(f"Model predictor: {model_predictor}\nGenotype architecture performance: {query_2}")

query_3, genotype = NAS301.query_bench(ind, model_predictor=model_predictor, returnGenotype=True)
print(f"Genotype: {genotype}\nModel predictor: {model_predictor}\nGenotype architecture performance: {query_3}")

"""
NASBench ASR
"""
from NASBench import ASR 
ind = [6, 3, 7, 4, 6, 2, 6, 7, 4, 3, 7, 7, 2, 5, 4, 1, 7, 5, 1, 4, 0, 5]
query_1 = ASR.query_bench(ind, dataset='cifar100')
query_2 = ASR.query_bench(ind, dataset='cifar100', metric='average_hw_metric')
print(query_1)
print(query_2)

"""
Reconstruct model from NASBench
"""
