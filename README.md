<div align='center'>

# NASBench

</div>

## Table of Contents
1. [NASBench](#1-nasbench)
        - NASBench101
        - NATS Bench
        - NASBench301
        - NASBench ASR
2. [Reconstruct model from NASBench](#2-reconstruct-model-from-nasbench)

-----
### 1. NASBench:
- **NASBench101**
        - Setup:
        ```bash
        cd source  
        gdown --fuzzy https://drive.google.com/uc\?id\=1D6IeM2cX-jrBhzuZGyMCD-emEXm6ndDW
        unzip nasbench.zip
        ```
        - Query:
        ```python
        from NASBench import NAS101

        matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
                [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
                [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
                [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
                [0, 0, 0, 0, 0, 0, 1],    # max-pool 3x3
                [0, 0, 0, 0, 0, 0, 0]]    # output layer

        query_1 = NAS101.query_bench(matrix) 
        print(f"Query 1:\n {query_1}\n")

        query_2 = NAS101.query_bench(matrix, metric='test_accuracy')
        print(f"Query 2:\n {query_2}")
        ```
- **NATS Bench**
        - Setup:
        ```bash
        cd source
        gdown --fuzzy https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view
        tar -xvf NATS-tss-v1_0-3ffb9-simple.tar
        ```

        - Query: 
        ```python
        from NASBench import NATS

        ind = [3, 3, 3, 3, 3, 3]

        query_1 = NATS.query_bench(ind, dataset='ImageNet16-120', epoch=200)
        print(f"Query 1:\n {query_1}\n")

        query_2 = NATS.query_bench(ind, dataset='ImageNet16-120', epoch=200, metric='test-accuracy')
        print(f"Query 2:\n {query_2}")
        ```
- **NASBench 301**
        - Setup:
        ```bash
        cd source && /bin/sh setup301.sh
        ```
        - Query:
        ```python
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
        ```

- **NASBench ASR**
        - Setup:
        ```bash
        cd source 
        git clone https://github.com/RICE-EIC/HW-NAS-Bench.git
        mv HW-NAS-Bench NASBenchHW
        ```
        - Query:
        ```python
        from NASBench import ASR 

        ind = [6, 3, 7, 4, 6, 2, 6, 7, 4, 3, 7, 7, 2, 5, 4, 1, 7, 5, 1, 4, 0, 5]

        query_1 = ASR.query_bench(ind, dataset='ImageNet')
        print(f"Query 1:\n {query_1}\n")

        query_2 = ASR.query_bench(ind, dataset='ImageNet' metric='average_hw_metric')
        print(f"Query 2:\n {query_2}")
        ```
### 2. Reconstruct model from NASBenchmark
- **NATS Bench**
```python
from NASBench import NATS
from ZeroCostNas.foresight.models.nasbench2 import get_model_from_arch_str

ind = [4, 0, 2, 2, 4, 3]
arch_str = NATS.convert_individual_to_str(ind)
model = get_model_from_arch_str(arch_str, num_classes=10)
```

- **NASBench 101**
```python

```