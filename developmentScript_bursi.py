#! /usr/bin/python
from neighborsMinHash import MinHash
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import time
import math
from collections import Counter
from eden.converter.graph.gspan import gspan_to_eden
from eden.graph import Vectorizer
import gc
graphs = gspan_to_eden( 'http://www.bioinf.uni-freiburg.de/~costa/bursi.gspan' )
vectorizer = Vectorizer( r=2,d=5 )
datasetBursi = vectorizer.transform( graphs )

n_neighbors_minHash = MinHash(number_of_hash_functions=400)
n_neighbors_minHash.fit(datasetBursi)

distribution = n_neighbors_minHash.get_distribution_of_inverse_index()
min_ = min(distribution[0])
max_ = max(distribution[0])

n_neighbors_sklearn = NearestNeighbors()
n_neighbors_sklearn.fit(datasetBursi)
neighbors_sklearn = n_neighbors_sklearn.kneighbors(return_distance=False)
error_list = []
time_fit_list = []
time_kneighbors_list = []
iterations = 3
distribution_inverse_index = []
for value in sorted(distribution[1][0::10]):
    time_fit_ = 0
    error_ = 0
    time_kneighbors_ = 0
    distribution_inverse_index_ = []
    for i in xrange(iterations):
        print "Iteration: ", i 
        n_neighbors_minHash_prune = MinHash(number_of_hash_functions=400,
                                        removeHashFunctionWithLessEntriesAs=value)
        time_start = time.time()
        n_neighbors_minHash_prune.fit(datasetBursi)
        distribution_inverse_index_.append(n_neighbors_minHash_prune.get_distribution_of_inverse_index())
        time_end = time.time() - time_start
        time_fit_ += time_end
        time_start = time.time()
        neighbors = n_neighbors_minHash_prune.kneighbors(return_distance=False, fast=False)
        time_end = time.time() - time_start
        time_kneighbors_ += time_end
        accuracy_value = 0
        for x, y in zip(neighbors, neighbors_sklearn):
            accuracy_value += accuracy_score(x, y)
        error_ += 1 - (accuracy_value / len(neighbors))
        # print "why: ", gc.get_referrers(n_neighbors_minHash_prune)
        
        del neighbors
        del n_neighbors_minHash_prune
        # print "gc.garbage: ", gc.garbage
        # gc.collect()
        # %xdel -n neighbors
        # %xdel -n n_neighbors_minHash_prune
#         del n_neighbors_minHash_prune
    time_fit_list.append(time_fit_ / iterations)
    time_kneighbors_list.append(time_kneighbors_ / iterations)
    error_list.append(error_ / iterations)
    dict_for_memory = {}
    for i in xrange(iterations):
        dict_for_memory = dict(Counter(dict_for_memory)+Counter(distribution_inverse_index_[i][0]))
    dict_for_memory.update((x, y / iterations) for x, y in dict_for_memory.items())
    distribution_inverse_index_ = None
    gc.collect()
    
    distribution_inverse_index.append([dict_for_memory])