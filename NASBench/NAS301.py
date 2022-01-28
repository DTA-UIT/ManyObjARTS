import os, sys
from collections import namedtuple
import torch
import numpy as np 
from ConfigSpace.read_and_write import json as cs_json
import nasbench301 as nb
from graphviz import Digraph
from random import choice

version = '1.0'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
url = os.path.dirname(__file__)
current_dir = f"{url[:-len('/NASBench')] + '/source/nasbench301/'}"
             
models_0_9_dir = os.path.join(current_dir, 'nb_models_0.9')
model_paths_0_9 = {
    model_name : os.path.join(models_0_9_dir, '{}_v0.9'.format(model_name))
    for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
}
models_1_0_dir = os.path.join(current_dir, 'nb_models_1.0')
model_paths_1_0 = {
    model_name : os.path.join(models_1_0_dir, '{}_v1.0'.format(model_name))
    for model_name in ['xgb', 'gnn_gin', 'lgb_runtime']
}

model_paths = model_paths_0_9 if version == '0.9' else model_paths_1_0

if not all(os.path.exists(model) for model in model_paths.values()):
    nb.download_models(version=version, delete_zip=True, download_dir=current_dir)

model_predictor = 'lgb_runtime'
print("==> Loading performance surrogate model...")
ensemble_dir_performance = model_paths[model_predictor]
print(ensemble_dir_performance)

performance_model = nb.load_ensemble(ensemble_dir_performance)

def random_connection( num_individuals ):
    connection0 = np.random.randint(2, size=(num_individuals, 2))
    connection1 = np.random.randint(3, size=(num_individuals, 2))
    connection2 = np.random.randint(4, size=(num_individuals, 2))
    connection3 = np.random.randint(5, size=(num_individuals, 2))
    connection = np.concatenate((connection0, connection1, connection2, connection3), axis=1)
    return connection

def repair_ind ( ind ):
    num = 1
    for i in range(9, len(ind) // 2, 2):
        num += 1
        if (ind[i] == ind[i - 1]):
            exclude_value = ind[i - 1]
            ind[i] = choice([j for j in range(num) if j not in [exclude_value]])
        
        if (ind[i + 16] == ind[i + 15]):
            exclude_value = ind[i + 15]
            ind[i + 16] = choice([j for j in range(num) if j not in [exclude_value]])
            
    return ind

choices = ["max_pool_3x3",
        "avg_pool_3x3",
        "skip_connect",
        "sep_conv_3x3",
        "sep_conv_5x5",
        "dil_conv_3x3",
        "dil_conv_5x5"]

def convert_ind_query(ind):
    normals = []
    reduces = []
    for i in range(len(ind) // 4):
        element_normal = (choices[ind[i]], ind[i + 8])
        normals.append(element_normal)
        element_reduce = (choices[ind[i + 16]], ind[i + 24])
        reduces.append(element_reduce)
    
    Genotype = namedtuple('Genotype', 
                          'normal normal_concat reduce reduce_concat')
    genotype_config = Genotype(
        normal=normals,
        normal_concat=list(range(2, 6)),
        reduce=reduces,
        reduce_concat=list(range(2, 6))
    )
    return genotype_config



def query_bench(ind, returnGenotype=False):
    genotype_config = convert_ind_query(repair_ind(ind))
    prediction_genotype = performance_model.predict(config=genotype_config, representation="genotype", with_noise=False)
    return prediction_genotype if not returnGenotype else (prediction_genotype, genotype_config)