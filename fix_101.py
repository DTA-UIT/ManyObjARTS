from NASBench import NAS101

api = NAS101.NAS101()
valid_ind = [[0, 1, 1, 0, 0, 1, 0,],
            [0, 0, 1, 0, 1, 1, 1,],
            [0, 0, 0, 1, 1, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 1, 0,],
            [0, 0, 0, 0, 0, 0, 0,],
            [0, 0, 0, 0, 0, 0, 0,]]

invalid_ind = [[0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,],
                [0, 0, 0, 0, 0, 0, 0,]] 
# api.query_bench(invalid_ind)
api.repair_connection(valid_ind)
# print(api.repair_connection(ind = [[0, 0, 0, 0, 0, 0, 0,],
#                         [0, 0, 0, 0, 0, 0, 0,],
#                         [0, 0, 0, 0, 0, 0, 0,],
#                         [0, 0, 0, 0, 0, 0, 0,],
#                         [0, 0, 0, 0, 0, 0, 0,],
#                         [0, 0, 0, 0, 0, 0, 0,],
#                         [0, 0, 0, 0, 0, 0, 0,]]))

