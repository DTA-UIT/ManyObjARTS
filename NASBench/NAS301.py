import os, sys
from collections import namedtuple
import torch
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