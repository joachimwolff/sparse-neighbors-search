# Copyright 2015 Joachim Wolff
# Master Thesis
# Tutor: Fabrizio Costa
# Winter semester 2015/2016
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwig-University Freiburg im Breisgau

from sklearn.cluster import DBSCAN
from ..neighbors import MinHash

import numpy as np

class MinHashDBSCAN():
    def __init__(self, eps=0.5, min_samples=5, 
        algorithm='auto', leaf_size=30, p=None, random_state=None, 
        fast=False, n_neighbors=5, radius=1.0,
        number_of_hash_functions=400,
        max_bin_size = 50, minimal_blocks_in_common = 1,
        shingle_size = 4, excess_factor = 5,
        number_of_cores=None, chunk_size=None):

        self.eps = eps
        self.min_samples = min_samples
        # self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.random_state = random_state
        self.radius = radius
        self.fast = fast
        self.number_of_hash_functions = number_of_hash_functions
        self.max_bin_size = max_bin_size
        self.minimal_blocks_in_common = minimal_blocks_in_common
        self.shingle_size = shingle_size
        self.excess_factor = excess_factor
        self.number_of_cores = number_of_cores
        self.chunk_size = chunk_size
        self.n_neighbors = n_neighbors

        self._dbscan = DBSCAN(eps=self.eps, min_samples=min_samples, metric='precomputed',
                algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.p, random_state=self.random_state)
        self.labels_ = None
        # only for compatible issues
    def fit(self, X, y=None):
        minHashNeighbors = MinHash(n_neighbors = self.n_neighbors, 
        radius = self.radius, fast = self.fast,
        number_of_hash_functions = self.number_of_hash_functions,
        max_bin_size = self.max_bin_size,
        minimal_blocks_in_common = self.minimal_blocks_in_common,
        shingle_size = self.shingle_size,
        excess_factor = self.excess_factor,
        number_of_cores = self.number_of_cores,
        chunk_size = self.chunk_size, similarity=False)
        minHashNeighbors.fit(X, y)
        graph_result = minHashNeighbors.kneighbors_graph(mode='distance')
        self._dbscan.fit(graph_result)
        self.labels_ = self._dbscan.labels_
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_