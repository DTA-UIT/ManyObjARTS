# """
# NASBench 101
# """
# from NASBench import NAS101
# matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
#         [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
#         [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
#         [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
#         [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
#         [0, 0, 0, 0, 0, 0, 1],    # max-pool 3x3
#         [0, 0, 0, 0, 0, 0, 0]]    # output layer
# api = NAS101.NAS101(ind=matrix)
# query_1 = api.query_bench() 
# query_2 = api.query_bench(metric='test_accuracy')
# print(f'Query 1: {query_1}\n')
# print(f'Query 2: {query_2}')

# """
# NATS Benchmark
# """
# from NASBench import NATS
# ind = [3, 3, 3, 3, 3, 3]
# api = NATS.NATS(ind=ind)
# query_1 = api.query_bench(dataset='cifar10', epoch=200)
# query_2 = api.query_bench(dataset='cifar10', epoch=200, metric='test-accuracy')
# print(f'Query 1: {query_1}\n')
# print(f'Query 2: {query_2}')

"""
NASBench 301
"""
from NASBench import NAS301

ind = [6, 3, 4, 6, 2, 4, 4, 6, 0, 0, 2, 1, 2, 0, 3, 0, 1, 3, 3, 6, 3, 6,
        3, 4, 0, 1, 1, 0, 3, 3, 0, 0]

query_1 = NAS301.query_bench(ind)
print(f"Query 1:\nGenotype architecture performance: {query_1}\n")

model_predictor = 'xgb'
query_2 = NAS301.query_bench(ind, model_predictor=model_predictor)
print(f"Query 2:\nModel predictor: {model_predictor}\nGenotype architecture performance: {query_2}\n")

query_3, genotype = NAS301.query_bench(ind, model_predictor=model_predictor, returnGenotype=True)
print(f"Query 3:\nGenotype: {genotype}\nModel predictor: {model_predictor}\nGenotype architecture performance: {query_3}\n")

# """
# NASBench ASR
# """
# from NASBench import ASR 
# ind = [6, 3, 7, 4, 6, 2, 6, 7, 4, 3, 7, 7, 2, 5, 4, 1, 7, 5, 1, 4, 0, 5]
# api = ASR.ASR(ind)
# query_1 = api.query_bench(dataset='cifar100')
# query_2 = api.query_bench(dataset='cifar100', metric='average_hw_metric')
# print(f'Query 1: {query_1}\n')
# print(f'Query 2: {query_2}')


# """
# Reconstruct model from NASBench
# """
