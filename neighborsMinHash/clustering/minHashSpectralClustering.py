# Copyright 2015 Joachim Wolff
# Master Thesis
# Tutors: Milad Miladi, Fabrizio Costa
# Winter semester 2015/2016
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwigs-University Freiburg im Breisgau

from ..neighbors import MinHash

from sklearn.cluster import SpectralClustering
import numpy as np

class MinHashSpectralClustering():
    def __init__(self, n_clusters=8, eigen_solver=None, 
                random_state=None, n_init=10, gamma=1.0, 
                n_neighbors=5, eigen_tol=0.0, 
                assign_labels='kmeans', degree=3, coef0=1, 
                kernel_params=None, 
                radius=1.0, fast=False, 
                number_of_hash_functions=400,
                max_bin_size = 50, minimal_blocks_in_common = 1,
                block_size = 4, excess_factor = 5,
                number_of_cores=None, chunk_size=None):
        self.n_clusters = n_clusters 
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.n_init = n_init
        self.gamma =  gamma
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0 
        self.kernel_params = kernel_params 
        self.radius = radius
        self.fast = fast
        self.number_of_hash_functions = number_of_hash_functions
        self.max_bin_size = max_bin_size
        self.minimal_blocks_in_common = minimal_blocks_in_common
        self.block_size = block_size
        self.excess_factor = excess_factor
        self.number_of_cores = number_of_cores
        self.chunk_size = chunk_size
        self._spectralClustering = SpectralClustering(n_clusters = self.n_clusters, 
                                                eigen_solver = self.eigen_solver,
                                                random_state = self.random_state,
                                                n_init = self.n_init,
                                                gamma =  self.gamma,
                                                affinity = 'precomputed',
                                                n_neighbors = self.n_neighbors,
                                                eigen_tol = self.eigen_tol,
                                                assign_labels = self.assign_labels,
                                                degree = self.degree,
                                                coef0 = self.coef0 ,
                                                kernel_params = self.kernel_params)
        # only for compatible issues
        self.labels_ = None
    def fit(self, X, y=None):
        minHashNeighbors = MinHash(n_neighbors = self.n_neighbors, 
        radius = self.radius, fast = self.fast,
        number_of_hash_functions = self.number_of_hash_functions,
        max_bin_size = self.max_bin_size,
        minimal_blocks_in_common = self.minimal_blocks_in_common,
        block_size = self.block_size,
        excess_factor = self.excess_factor,
        number_of_cores = self.number_of_cores,
        chunk_size = self.chunk_size, similarity=True)
        minHashNeighbors.fit(X, y)
        graph_result = minHashNeighbors.kneighbors_graph(mode='distance')
        self._spectralClustering.fit(graph_result)
        self.labels_ = self._spectralClustering.labels_
    def fit_predict(self, X, y=None):
        self.fit(X, y)

        return self._spectralClustering.labels_