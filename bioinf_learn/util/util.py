# Copyright 2016 Joachim Wolff
# Master Project
# Tutors: Milad Miladi, Fabrizio Costa
# Summer semester 2015
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwigs-University Freiburg im Breisgau

__author__ = 'joachimwolff'

from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
#import neighbors as kneighbors
import random
import time
import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse import rand
from scipy.sparse import vstack

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LSHForest
from sklearn.metrics import accuracy_score
import sklearn
from sklearn.random_projection import SparseRandomProjection
from ..neighbors import MinHash
# import pyflann
import annoy

import matplotlib.pyplot as plt


def neighborhood_accuracy(neighbors, neighbors_sklearn):
    """Computes the accuracy for the exact and approximate version of the MinHash algorithm.

    Parameters
    ----------
    neighbors_exact : array[[neighbors]]
    neighbors_sklearn : array[[neighbors]]

    Returns
    -------
    exact_accuracy : float
        The accuracy between the exact version of the algorithm and the nearestNeighbors implementation of sklearn.
    approximate_accuracy : float
        The accuracy between the approximate version of the algorithm and the nearestNeighbors implementation of sklearn.
    approx_exact : float
        The accuracy between the approximate and the exact version of the algorithm.
    """
    accuracy = 0;
    for i in xrange(len(neighbors)):
        accuracy += len(np.intersect1d(neighbors[i], neighbors_sklearn[i]))
    
    return accuracy / float(len(neighbors) * len(neighbors[0]))
    

def create_dataset(seed=None,
                   number_of_centroids=None,
                   number_of_instances=None,
                   number_of_features=None,
                   size_of_dataset=None,
                   density=None,
                   fraction_of_density=None
                   ):
    """Create an artificial sparse dataset
    Prameters
    ---------
    seed : int
        Random seed
    number_of_centroids : int
        How many centroids the created dataset should have. 
    number_of_instances : int
        How many instances the whole dataset, noise included, should have. 
        It should hold: number_of_centroids * size_of_dataset <= number_of_instances. The difference between 
        number_of_centroids * size_of_dataset and number_of_instances is filled up with noise.
    number_of_features : int
        How many features each instance should have.
    size_of_dataset : int
        How many instances each cluster defined by a centroid should have.
    density : float 
        The sparsity of the dataset
    fraction_of_density : float
        How much noise each cluster should have inside.

     Returns
    -------

    X : sparse csr_matrix [instances, features]
    y : array with classes [classes]

    """
    dataset_neighborhood_list = []
    number_of_swapping_elements = int(number_of_features * density * fraction_of_density)
    y = []
    random_local = random.randint
    number_of_features_density = int(number_of_features*density)-1
    for k in xrange(number_of_centroids):
        dataset_neighbor = rand(1, number_of_features, density=density, format='lil', random_state=seed*k)
        nonzero_elements =  dataset_neighbor.nonzero()[1]
        for i in xrange(size_of_dataset):
            neighbor = dataset_neighbor.copy()
            # random.seed(seed*k)
            for j in xrange(number_of_swapping_elements):
                index = random_local(0, number_of_features_density)
                index_swap = random_local(0, number_of_features-1)
                neighbor[0, nonzero_elements[index]] = 0
                neighbor[0, index_swap] = 1
            dataset_neighborhood_list.append(neighbor)
        y.append(k)

    dataset_neighborhood = vstack(dataset_neighborhood_list)

    size_of_noise = number_of_instances-(number_of_centroids*size_of_dataset)
    if size_of_noise > 0:
            dataset_noise = rand(size_of_noise, number_of_features, format='lil', density=density, random_state=seed*seed)
            dataset = vstack([dataset_neighborhood, dataset_noise])
    else:
        dataset = vstack([dataset_neighborhood])
    random_value_generator = random.randint

    # add classes for noisy data
    for i in range(0, size_of_noise):
        y.append(random_value_generator(0, number_of_centroids))

    return csr_matrix(dataset), y


def create_dataset_fixed_nonzero(seed=None,
                   number_of_centroids=None,
                   number_of_instances=None,
                   number_of_features=None,
                   size_of_dataset=None,
                   non_zero_elements=None,
                   fraction_of_density=None):
    """Create an artificial sparse dataset with a fixed number of nonzero elements.

    Prameters
    ---------
    seed : int
        Random seed
    number_of_centroids : int
        How many centroids the created dataset should have. 
    number_of_instances : int
        How many instances the whole dataset, noise included, should have. 
        It should hold: number_of_centroids * size_of_dataset <= number_of_instances. The difference between 
        number_of_centroids * size_of_dataset and number_of_instances is filled up with noise.
    number_of_features : int
        How many features each instance should have.
    size_of_dataset : int
        How many instances each cluster defined by a centroid should have.
    fraction_of_density : float
        How much noise each cluster should have inside.

    Returns
    -------

    X : sparse csr_matrix [instances, features]
    y : array with classes [classes]
    """
    if (non_zero_elements > number_of_features):
        print "More non-zero elements than features!"
        return
    density = non_zero_elements / float(number_of_features)
    print "Desity:" , density
    dataset_neighborhood_list = []
    number_of_swapping_elements = int(non_zero_elements * fraction_of_density)
    y = []
    random_local = random.randint
    
    for k in xrange(number_of_centroids):
        dataset_neighbor = rand(1, number_of_features, density=density, format='lil', random_state=seed*k)
        nonzero_elements =  dataset_neighbor.nonzero()[1]
        for i in xrange(size_of_dataset):
            neighbor = dataset_neighbor.copy()
            # random.seed(seed*k)
            for j in xrange(number_of_swapping_elements):
                index = random_local(0, non_zero_elements-1)
                index_swap = random_local(0, number_of_features-1)
                neighbor[0, nonzero_elements[index]] = 0
                neighbor[0, index_swap] = 1
            dataset_neighborhood_list.append(neighbor)
        y.append(k)

    dataset_neighborhood = vstack(dataset_neighborhood_list)

    size_of_noise = number_of_instances-(number_of_centroids*size_of_dataset)
    if size_of_noise > 0:
            dataset_noise = rand(size_of_noise, number_of_features, format='lil', density=density, random_state=seed*seed)
            dataset = vstack([dataset_neighborhood, dataset_noise])
    else:
        dataset = vstack([dataset_neighborhood])
    random_value_generator = random.randint

    # add classes for noisy data
    for i in range(0, size_of_noise):
        y.append(random_value_generator(0, number_of_centroids))

    return csr_matrix(dataset), y


def benchmarkNearestNeighorAlgorithms(dataset, n_neighbors = 10, reduce_dimensions_to=400):
    """Function to measure the performance for the given input data.
        If the nearest neighbor algorithm does not support sparse datasets
        a random projection is used to create a dense dataset.
        """
    
    import sklearn.neighbors
    import pyflann
    import annoy
    import panns
    import nearpy, nearpy.hashes, nearpy.distances
    import pykgraph
    import nmslib
    from rpforest import RPForest
    accuracy_minHash = 0
    accuracy_minHashFast = 0
    accuracy_lshf = 0
    accuracy_annoy = 0
    accuracy_ballTree = 0
    accuracy_KDTree = 0
    accuracy_FLANN = 0
    accuracy_PANNS = 0
    accuracy_NearPy = 0
    accuracy_KGraph = 0
    accuracy_Nmslib = 0
    accuracy_RPForest = 0
    
    time_fitting_bruteforce = 0
    time_fitting_minHash = 0
    time_fitting_minHashFast = 0
    time_fitting_lshf = 0
    time_fitting_annoy = 0
    time_fitting_ballTree = 0
    time_fitting_KDTree = 0
    time_fitting_FLANN = 0
    time_fitting_PANNS = 0
    time_fitting_NearPy = 0
    time_fitting_KGraph = 0
    time_fitting_Nmslib = 0
    time_fitting_RPForest = 0
    
    time_query_bruteforce = 0
    time_query_minHash = 0
    time_query_minHashFast = 0
    time_query_lshf = 0
    time_query_annoy = 0
    time_query_ballTree = 0
    time_query_KDTree = 0
    time_query_FLANN = 0
    time_query_PANNS = 0
    time_query_NearPy = 0
    time_query_KGraph = 0
    time_query_Nmslib = 0
    time_query_RPForest = 0
        
    data_projection = SparseRandomProjection(n_components=reduce_dimensions_to, random_state=1)
    dataset_dense = data_projection.fit_transform(dataset)
    
    dataset_dense = sklearn.preprocessing.normalize(dataset_dense, axis=1, norm='l2')
    
    # create object and fit
    
    # brute force
    time_start = time.time()
    brute_force_obj = NearestNeighbors(n_neighbors = n_neighbors)
    brute_force_obj.fit(dataset)
    time_fitting_bruteforce = time.time() - time_start
    
    # minHash
    time_start = time.time(dataset)
    minHash_obj = MinHash(n_neighbors = n_neighbors, number_of_hash_functions=hash_function,
                         max_bin_size= 36, shingle_size = 4, similarity=False, 
                    prune_inverse_index=2, store_value_with_least_sigificant_bit=3, excess_factor=11,
                    prune_inverse_index_after_instance=0.5, remove_hash_function_with_less_entries_as=4, 
                    hash_algorithm = 0, shingle=0, block_size=2, cpu_gpu_load_balancing = 1.0)
    minHash_obj.fit(dataset)
    time_fitting_minHash = time.time() - time_start
    
    # minHash fast
    time_start = time.time()
    minHash_fast_obj = MinHash(n_neighbors = n_neighbors, number_of_hash_functions=hash_function,
                         max_bin_size= 36, shingle_size = 4, similarity=False, fast=True,
                    prune_inverse_index=2, store_value_with_least_sigificant_bit=3, excess_factor=11,
                    prune_inverse_index_after_instance=0.5, remove_hash_function_with_less_entries_as=4, 
                    hash_algorithm = 0, shingle=0, block_size=2, cpu_gpu_load_balancing = 1.0)      
    minHash_fast_obj.fit(dataset)
    time_fitting_minHashFast = time.time() - time_start
    
    # lshf
    time_start = time.time()                       
    lshf_obj = LSHForest(n_estimators=20, n_candidates=200, n_neighbors=n_neighbors)
    lshf_obj.fit(dataset_dense)
    time_fitting_lshf = time.time() - time_start
    
    # ball tree
    time_start = time.time()
    ballTree_obj = sklearn.neighbors.BallTree(X, leaf_size=self._leaf_size)
    time_fitting_ballTree = time.time() - time_start
    
    
    # kd tree
    time_start = time.time()
    kdTree_obj = sklearn.neighbors.KDTree(X, leaf_size=self._leaf_size)
    time_fitting_KDTree = time.time() - time_start
    
    # flann
    time_start = time.time()
    flann_obj = pyflann.FLANN(target_precision=self._target_precision, algorithm='autotuned', log_level='info')
    flann.build_index(X)
    
    # annoy
    time_start = time.time()    
    annoy_obj = annoy.AnnoyIndex(f=X.shape[1], metric=self._metric)
    for i, x in enumerate(X):
        self._annoy.add_item(i, x.tolist())
    self._annoy.build(self._n_trees)
    time_fitting_annoy = time.time() - time_start
    
    # panns    
    time_start = time.time()
    panns_obj = panns.PannsIndex(X.shape[1], metric=self._metric)
    for x in X:
        panns_obj.add_vector(x)
    panns_obj.build(n_trees)
    time_fitting_PANNS = time.time() - time_start
    
    # nearpy
    time_start = time.time()
    nearpy_obj = 
    
    time_start = time.time()
    time_start = time.time()
    time_start = time.time()
    time_start = time.time()
    time_start = time.time()
    accuracy_list = [accuracy_minHash, accuracy_minHashFast, accuracy_lshf, accuracy_annoy, accuracy_ballTree,
                        accuracy_KDTree, accuracy_FLANN, accuracy_PANNS, accuracy_NearPy, accuracy_KGraph, 
                        accuracy_Nmslib, accuracy_RPForest]
    time_fitting_list = [time_fitting_bruteforce, time_fitting_minHash, time_fitting_minHashFast, time_fitting_lshf,
                            time_fitting_annoy, time_fitting_ballTree, time_fitting_KDTree, time_fitting_FLANN,
                            time_fitting_PANNS, time_fitting_NearPy, time_fitting_KGraph, time_fitting_Nmslib,
                            time_fitting_RPForest]
    time_query_list = [time_query_bruteforce, time_query_minHash, time_query_minHashFast, time_query_lshf,
                        time_query_annoy, time_query_ballTree, time_query_KDTree, time_query_FLANN,
                        time_query_PANNS, time_query_NearPy, time_query_KGraph, time_query_Nmslib,
                        time_query_RPForest]
    return [accuracy_list, time_fitting_list, time_query_list]

def plotData(data, color, label, title, xticks, ylabel,
         number_of_instances, number_of_features,
         figure_size=(10,5),  bar_width=0.1,log=True, xlabel=None):
    plt.figure(figsize=figure_size)
    N = number_of_instances * number_of_features

    ind = np.arange(N)    # the x locations for the groups
    
    #"r", "b", "g", "c", "m", "y", "k", "w"
    count = 0
    for d, c, l in zip(data, color, label):
        plt.bar(ind + count * bar_width , d,   bar_width, color=c, label=l)
        count += 1
    if log:
        plt.yscale('log')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.xticks(ind+3*bar_width, (xticks))
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True)
    plt.show()

def plotDataBenchmark(data, color, label, title, xticks, ylabel,
         number_of_instances, number_of_features,
         figure_size=(10,5),  bar_width=0.1,log=True, xlabel=None):
    plt.figure(figsize=figure_size)
    N = number_of_instances * number_of_features

    ind = np.arange(N)    # the x locations for the groups
    
    #"r", "b", "g", "c", "m", "y", "k", "w"
    count = 0
    for d, c, l in zip(data, color, label):
        plt.bar(ind + count * bar_width , d,   bar_width, color=c, label=l)
        count += 1
    if log:
        plt.yscale('log')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.xticks(ind+3*bar_width, (xticks))
    plt.legend(loc='upper left', fontsize='small')
    plt.grid(True)
    plt.show()