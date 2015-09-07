# Copyright 2015 Joachim Wolff
# Master Project
# Tutors: Milad Miladi, Fabrizio Costa
# Summer semester 2015
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwig-University Freiburg im Breisgau

from ..neighbors import MinHashNearestNeighbors

from sklearn.cluster import SpectralClustering

class MinHashSpectralClustering():
    def __init__(self, n_clusters=8, eigen_solver=None, 
                random_state=None, n_init=10, gamma=1.0, 
                affinity='rbf', n_neighbors=10, eigen_tol=0.0, 
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
        self.affinity = affinity
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

    def fit(self, X, y=None):
        minHashNeighbors = MinHashNearestNeighbors(n_neighbors = self.n_neighbors, 
        radius = self.radius, fast = self.fast,
        number_of_hash_functions = self.number_of_hash_functions,
        max_bin_size = self.max_bin_size,
        minimal_blocks_in_common = self.minimal_blocks_in_common,
        block_size = self.block_size,
        excess_factor = self.excess_factor,
        number_of_cores = self.number_of_cores,
        chunk_size = self.chunk_size)
        minHashNeighbors.fit(X, y)
        graph_result = minHashNeighbors.kneighbors_graph()
        spectralClustering = SpectralClustering(n_clusters = self.n_clusters, 
                                                eigen_solver = self.eigen_solver,
                                                random_state = self.random_state,
                                                n_init = self.n_init,
                                                gamma =  self.gamma,
                                                affinity = self.affinity,
                                                n_neighbors = self.n_neighbors,
                                                eigen_tol = self.eigen_tol,
                                                assign_labels = self.assign_labels,
                                                degree = self.degree,
                                                coef0 = self.coef0 ,
                                                kernel_params = self.kernel_params)
        spectralClustering.fit(graph_result)
    def fit_predict(X, y=None):
        pass 

