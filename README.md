<div align='center'>

# NASBench

</div>

More details on setting up and querying [here](https://colab.research.google.com/drive/1TWC0iosUhfdmNRIGW_VLw4MZfPwblWiF?authuser=1#scrollTo=u9shlcqGuzGk)

## Table of Contents
1. [First time set up](#1-first-time-set-up)
2. [Query using NASBench:](#2-query-using-nasbench)
3. [Reconstruct model from NASBenchmark](#3-reconstruct-model-from-nasbenchmark)

-----

###  1. First time set up
#### a. **NASBench101**
```bash
cd source  
gdown --fuzzy https://drive.google.com/uc\?id\=1D6IeM2cX-jrBhzuZGyMCD-emEXm6ndDW
unzip nasbench.zip
```
#### b. **NASBench ASR**
```bash
cd source 
git clone https://github.com/RICE-EIC/HW-NAS-Bench.git
mv HW-NAS-Bench NASBenchHW
```
#### c. **NATS Bench**: 
```bash
cd source
gdown --fuzzy https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view
tar -xvf NATS-tss-v1_0-3ffb9-simple.tar
```

#### d. **NASBench 301**
```bash
cd source && /bin/sh setup301.sh
```

### 2. Query using NASBench:
#### a. **NASBench101**
```python
from NASBench import NAS101

matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
        [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
        [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
        [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],    # max-pool 3x3
        [0, 0, 0, 0, 0, 0, 0]]    # output layer

api = NAS101.NAS101(ind=matrix)

query_1 = api.query_bench() 
print(f"Query 1:\n {query_1}\n")

query_2 = api.query_bench(metric='test_accuracy')
print(f"Query 2:\n {query_2}")
```

#### b. **NATS Bench**
```python
from NASBench import NATS

ind = [3, 3, 3, 3, 3, 3]

api = NATS.NATS(ind=ind)

query_1 = api.query_bench(dataset='cifar10', epoch=200)
query_2 = api.query_bench(dataset='cifar10', epoch=200, metric='test-accuracy')

print(f'Query 1: {query_1}\n')
print(f'Query 2: {query_2}')
```

#### c. **NASBench 301**
```python
from NASBench import NAS301

ind = [6, 3, 4, 6, 2, 4, 4, 6, 0, 0, 2, 1, 2, 0, 3, 0, 1, 3, 3, 6, 3, 6,
        3, 4, 0, 1, 1, 0, 3, 3, 0, 0]

api = NAS301.NAS301(ind)

query_1 = api.query_bench()
print(f"Query 1:\nGenotype architecture performance: {query_1}\n")

model_predictor = 'xgb'
query_2 = api.query_bench(model_predictor=model_predictor)
print(f"Query 2:\nModel predictor: {model_predictor}\nGenotype architecture performance: {query_2}\n")

model_predictor = 'xgb'
query_3, genotype = api.query_bench(model_predictor=model_predictor, returnGenotype=True)
print(f"Query 3:\nGenotype: {genotype}\nModel predictor: {model_predictor}\nGenotype architecture performance: {query_3}\n")
```

#### d. **NASBench ASR**
```python
from NASBench import ASR 

ind = [6, 3, 7, 4, 6, 2, 6, 7, 4, 3, 7, 7, 2, 5, 4, 1, 7, 5, 1, 4, 0, 5]

api = ASR.ASR(ind=ind)

query_1 = api.query_bench(dataset='cifar100')
print(f"Query 1:\n {query_1}\n")

query_2 = api.query_bench(dataset='cifar100', metric='average_hw_metric')
print(f"Query 2:\n {query_2}")
```
### 3. Reconstruct model from NASBenchmark
#### a. **NATS Bench**
```python
from ZeroCostNas.foresight.models.nasbench2 import get_model_from_arch_str
from NASBench import NATS

ind = [4, 0, 2, 2, 4, 3]
api = NATS.NATS(ind)

arch_str = api.convert_individual_to_query()
model = get_model_from_arch_str(arch_str, num_classes=10)

print(f'Architecture string: {arch_str}\n\n')
print(model)
```

#### b. **NASBench 101**
```python
from NASBench import NAS101
from ZeroCostNas.foresight.models import nasbench1

ind = [[0, 1, 1, 1, 0, 1, 0],    # input layer
    [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
    [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
    [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
    [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
    [0, 0, 0, 0, 0, 0, 1],    # max-pool 3x3
    [0, 0, 0, 0, 0, 0, 0]]    # output layer
    
api = NAS101.NAS101(ind)
cell = api.cell
model = nasbench1.Network(cell, stem_out=128, num_stacks=3, num_mods=3, num_classes=10)

print(f'Architecture: {cell}\n')
print(model)
```
