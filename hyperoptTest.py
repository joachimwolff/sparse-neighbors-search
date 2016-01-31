#! /usr/bin/python
from neighborsMinHash import MinHash
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import time
import math
from collections import Counter
import gc

import os.path
import cPickle as pickle
def compute_score(error_list, memory_list, time_list, max_memory, max_time, alpha, beta):
    score_values = []
    for error_, memory_, time_ in zip(error_list, memory_list, time_list):
        error = error_/(float (max_error))
        if memory_ == 0: 
            score_values.append(5)
            continue
        else:
            memory = math.log(memory_/float(max_memory), 10)*alpha
        time = math.log(time_/float(max_time), 10)*beta
        score_values.append(error + memory  + time)
    score_values_np = np.array(score_values)
    min_values_index = np.array(np.where(score_values_np == score_values_np.min()))
    min_values = score_values_np.min()
    return min_values, min_values_index, score_values

def getMemory(distribution_pruning, iteration_value_list, time_fit_list, error_list, time_kneighbors_list):
    memory_list = []
    for dist in distribution_pruning:
        memory = 0
        for j in dist[0][0]:
            memory += j * dist[0][0][j]
        memory_list.append(memory)
    tmp_memory = []
    tmp_index = []
    tmp_fitting_time = []
    tmp_query_time = []
    tmp_error_list = []
    relative_error_reduction_list = []
    for i in xrange(len(memory_list)):
        if memory_list[i] != 0:
            tmp_memory.append(memory_list[i])
            tmp_index.append(iteration_value_list[i])
            tmp_fitting_time.append(time_fit_list[i])
            tmp_error_list.append(error_list[i])
            tmp_query_time.append(time_kneighbors_list[i])
            value_relative_error = relative_error_reduction(max_error, error_list[i])
            relative_error_reduction_list.append(value_relative_error)
    memory_list = tmp_memory
    iteration_value_list = tmp_index
    time_fit_list = tmp_fitting_time
    error_list = tmp_error_list
    time_kneighbors_list = tmp_query_time
    return memory_list, iteration_value_list, time_fit_list, error_list, time_kneighbors_list, relative_error_reduction_list

def relative_error_reduction(max_error, new_error):
#     print max_error
#     print new_error
#     print "rel: ", (max_error - new_error)/float(max_error)
    return (max_error - new_error)/float(max_error)

input_file_name = "/home/joachim/thesis/minHashNearestNeighbors/result/init_values"
if os.path.isfile(input_file_name):
    fileObject = open(input_file_name,'r')
    distribution = pickle.load(fileObject)
    
min_ = min(distribution[0][0])
max_ = max(distribution[0][0])

max_error = 1 - distribution[3]
max_time = distribution[2]
memory_list = []
max_memory = 0
for i in distribution[0][0]:
    memory_list.append(distribution[0][0][i])
    max_memory += i*distribution[0][0][i]
    

error_list = []
time_fit_list = []
time_kneighbors_list = []
distribution_pruning = []
# prune_index_frequencey+${i}
index_shift = 1
index_values = []
frequency_pruning = [0.5, 0.7, 1]
iteration_value_list = []
iteration_value = []
file_obj = open("/home/joachim/thesis/minHashNearestNeighbors/result/size_of_hash_functions", 'r')
lines = file_obj.readlines()
for i in lines:
    iter_str = i.strip().split(" ")
    for i in iter_str:
        iteration_value.append(int(i))
file_obj.close()
iteration_value = sorted(iteration_value)
for i in [0, 1]:
    for j in [-1, 0.5]:
        for k in [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for l in iteration_value:
                for m in [4, 5]:
                    input_file_name = "/home/joachim/thesis/minHashNearestNeighbors/result/all+"+str(i)+ "+" +str(j)+ "+" +str(k)+ "+" +str(l)+"+" +str(m)
                    if os.path.isfile(input_file_name):
                        fileObject = open(input_file_name,'r')
                        dist_tmp = pickle.load(fileObject)
                        distribution_pruning.append(dist_tmp)
                        error_list.append(1 - dist_tmp[3])
                        time_fit_list.append(dist_tmp[1])
                        time_kneighbors_list.append(dist_tmp[2])
                        iteration_value_list.append([i,j,k,l,m])
                for m in [1, 2, 3]:
                    input_file_name = "/home/joachim/thesis/minHashNearestNeighbors/result/all+"+str(i)+ "+" +str(j)+ "+" +str(k)+ "+" +str(l)+"+" +str(m)
                    if os.path.isfile(input_file_name):
                        fileObject = open(input_file_name,'r')
                        dist_tmp = pickle.load(fileObject)
                        distribution_pruning.append(dist_tmp)
                        error_list.append(1 - dist_tmp[3])
                        time_fit_list.append(dist_tmp[1])
                        time_kneighbors_list.append(dist_tmp[2])
                        iteration_value_list.append([i,j,k,l,m])
                        
                
memory_list, iteration_value_list, time_fit_list, error_list, time_kneighbors_list, relative_error_reduction_list = \
getMemory(distribution_pruning, iteration_value_list, time_fit_list, error_list, time_kneighbors_list)


# define an objective function
def objective(args):
    case, val = args
    minHash = MinHash()
    return compute_score(error_list, memory_list, time_kneighbors_list, max_memory, max_time, val, val)[0]

# define a search space
from hyperopt import hp
space = hp.choice('parameters',
    [{
        'number_of_hash_functions': hp.uniform(50, 1000),
        'max_bin_size': hp.uniform(1, 100),
        'shingle_size': hp.uniform(1, 5),
        'excess_factor': hp.uniform(1, 20),
        'chunk_size': hp.uniform(1, 20),
        'prune_inverse_index': [hp.uniform(1, 20), -1],
        'prune_inverse_index_after_instance': 
            hp.choice('prune_inverse_index_after_instance',[ -1.0, hp.uniform(0.0, 1.0)]),
        
    }
        
    ])

# minimize the objective over the space
from hyperopt import fmin, tpe
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print best
# -> {'a': 1, 'c2': 0.01420615366247227}
print hyperopt.space_eval(space, best)
# -> ('case 2', 0.01420615366247227}