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


def benchmarkNearestNeighborAlgorithms(dataset, n_neighbors = 10, reduce_dimensions_to=400):
    """Function to measure the performance for the given input data.
        If the nearest neighbor algorithm does not support sparse datasets
        a random projection is used to create a dense dataset.
        """
    
    import sklearn.neighbors
    # import pyflann
    import annoy
    import panns
    import nearpy, nearpy.hashes, nearpy.distances
    # import pykgraph
    # import nmslib
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
                    prune_inverse_index_after_instance=0.5, store_hash_function_with_less_entries_as=4, 
                     shingle=0, block_size=2, cpu_gpu_load_balancing = 1.0)
    minHash_obj.fit(dataset)
    time_fitting_minHash = time.time() - time_start
    
    # minHash fast
    time_start = time.time()
    minHash_fast_obj = MinHash(n_neighbors = n_neighbors, number_of_hash_functions=hash_function,
                         max_bin_size= 36, shingle_size = 4, similarity=False, fast=True,
                    prune_inverse_index=2, store_value_with_least_sigificant_bit=3, excess_factor=11,
                    prune_inverse_index_after_instance=0.5, store_hash_function_with_less_entries_as=4, 
                     shingle=0, block_size=2, cpu_gpu_load_balancing = 1.0)      
    minHash_fast_obj.fit(dataset)
    time_fitting_minHashFast = time.time() - time_start
    
    # lshf
    time_start = time.time()                       
    lshf_obj = LSHForest(n_estimators=20, n_candidates=200, n_neighbors=n_neighbors)
    lshf_obj.fit(dataset_dense)
    time_fitting_lshf = time.time() - time_start
    
    # ball tree
    time_start = time.time()
    ballTree_obj = sklearn.neighbors.BallTree(dataset, leaf_size=20)
    time_fitting_ballTree = time.time() - time_start
    
    
    # kd tree
    time_start = time.time()
    kdTree_obj = sklearn.neighbors.KDTree(dataset, leaf_size=20)
    time_fitting_KDTree = time.time() - time_start
    
    # flann
    time_start = time.time()
    # flann_obj = pyflann.FLANN(target_precision=0.9, algorithm='autotuned', log_level='info')
    # flann.build_index(dataset)
    time_fitting_FLANN = time.time() - time_start
    # annoy
    time_start = time.time()    
    annoy_obj = annoy.AnnoyIndex(f=dataset.shape[1], metric='euclidean')
    for i, x in enumerate(X):
        self._annoy.add_item(i, x.tolist())
    self._annoy.build(100)
    time_fitting_annoy = time.time() - time_start
    
    # panns    
    time_start = time.time()
    panns_obj = panns.PannsIndex(dataset.shape[1], metric='euclidean')
    for x in dataset:
        panns_obj.add_vector(x)
    panns_obj.build(100)
    time_fitting_PANNS = time.time() - time_start
    
    # nearpy
    time_start = time.time()
    hashes = []
    hash_counts = 10
    n_bits = 10
    # TODO: doesn't seem like the NearPy code is using the metric??
    for k in xrange(hash_counts):
        nearpy_rbp = nearpy.hashes.RandomBinaryProjections('rbp_%d' % k, n_bits)
        hashes.append(nearpy_rbp)

    nearpy_obj = nearpy.Engine(dataset.shape[1], lshashes=hashes)

    for i, x in enumerate(dataset):
        nearpy_obj.store_vector(x.tolist(), i)
    time_fitting_NearPy = time.time() - time_start
    
    # kgraph
    time_start = time.time()
    kgraph_obj = pykgraph.KGraph()
    kgraph_obj.build(dataset, iterations=30, L=100, delta=0.002, recall=0.99, K=25)
    time_fitting_KGraph = time.time() - time_start
    
    #nmslib
    time_start = time.time()
    # Nmslib(m, 'lsh_multiprobe', ['desiredRecall=0.90','H=1200001','T=10','L=50','tuneK=10']),
    nmslib_obj = nmslib.initIndex(dataset.shape[0], 'l2', [], 'lsh_multiprobe', ['desiredRecall=0.90','H=1200001','T=10','L=50','tuneK=10'], nmslib.DataType.VECTOR, nmslib.DistType.FLOAT)
	
    for i, x in enumerate(dataset):
        nmslib.setData(nmslib_obj, i, x.tolist())
    nmslib.buildIndex(nmslib_obj)
    time_fitting_Nmslib = time.time() - time_start
    
    
    # rpforest
    time_start = time.time()
    rpforest_obj = RPForest(leaf_size=10, no_trees=100)
    rpforest_obj.fit(dataset)
    time_fitting_RPForest = time.time() - time_start
  
  
  
    # query part
    # brute force
    time_start = time.time()
    neighbors_bruteforce = brute_force_obj.kneighbors(n_neighbors=n_neighbors, return_distance=False)
    time_query_bruteforce = time.time() - time_start
    
    # minHash
    time_start = time.time()
    neighbors_minHash = minHash_obj.kneighbors(n_neighbors=n_neighbors, return_distance=False)
    time_query_minHash = time.time() - time_start
    accuracy_minHash = neighborhood_accuracy(neighbors_minHash, neighbors_bruteforce)
    # minHash fast
    time_start = time.time()
    neighbors_minHash_fast = minHash_fast_obj.kneighbors(n_neighbors=n_neighbors, return_distance=False)
    time_query_minHashFast = time.time() - time_start 
    accuracy_minHashFast = neighborhood_accuracy(neighbors_minHash_fast, neighbors_bruteforce)
    # lshf
    time_start = time.time()
    neighbors_lshf = lshf_obj.kneighbors(dataset_dense, n_neighbors=n_neighbors, return_distance=False)
    time_query_lshf = time.time() - time_start
    accuracy_lshf = neighborhood_accuracy(neighbors_lshf, neighbors_bruteforce)
    # annnoy
    time_start = time.time()
    neighbors_annoy = []
    for i in xrange(dataset_dense.shape[0]):
    # annoy_.get_nns_by_vector(dataset_dense.getrow(i).toarray(), n_neighbors_sklearn, 100)
        neighbors_annoy.append(annoy_obj.get_nns_by_vector(dataset_dense.getrow(i).toarray(), n_neighbors, 100))
    time_query_annoy = time.time() - time_start 
    accuracy_annoy = neighborhood_accuracy(neighbors_annoy, neighbors_bruteforce)
    # ball tree
    time_start = time.time()
    dist, ind = ballTree_obj.query(dataset, k=n_neighbors)
    neighbors_balltree = ind
    time_query_ballTree = time.time() - time_start
    accuracy_ballTree = neighborhood_accuracy(neighbors_balltree, neighbors_bruteforce)
    # kd tree
    time_start = time.time()
    neighbors_kdTree = kdTree_obj.query(dataset, k=n_neighbors)
    time_query_KDTree = time.time() - time_start
    accuracy_KDTree = neighborhood_accuracy(neighbors_kdTree, neighbors_bruteforce)
    # flann
    time_start = time.time()
    # neighbors_flann = flann_obj.nn_index(dataset, n_neighbors)
    time_query_FLANN  = time.time() - time_start
    # accuracy_FLANN = neighborhood_accuracy(neighbors_flann, neighbors_bruteforce)
    # panns
    time_start = time.time()
    neigbors_panns = panns_obj.query(dataset, n_neighbors)
    time_query_PANNS  = time.time() - time_start
    accuracy_PANNS = neighborhood_accuracy(neigbors_panns, neighbors_bruteforce)
    # nearpy
    time_start = time.time()
    neighbors_nearpy = nearpy_obj.neighbours(dataset)
    time_query_NearPy  = time.time() - time_start
    accuracy_NearPy = neighborhood_accuracy(neighbors_nearpy, neighbors_bruteforce)
    # kgraph
    time_start = time.time()
    neighbors_kgraph = kgraph_obj.search(dataset, dataset, K=n_neighbors, threads=4, P=200)
    time_query_KGraph  = time.time() - time_start
    accuracy_KGraph = neighborhood_accuracy(neighbors_kgraph, neighbors_bruteforce)
    # nmslib
    time_start = time.time()
    neighbors_nmslib = nmslib.knnQuery(nmslib_obj, n_neighbors, dataset)
    time_query_Nmslib  = time.time() - time_start
    accuracy_Nmslib = neighborhood_accuracy(neighbors_nmslib, neighbors_bruteforce)
    # rpforest
    time_start = time.time()
    neighbors_rpforest = rpforest_obj.query(dataset, n_neighbors)
    time_query_RPForest  = time.time() - time_start
    accuracy_RPForest = neighborhood_accuracy(neighbors_rpforest, neighbors_bruteforce)
    
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

def measure_performance(dataset, n_neighbors_sklearn = 10, n_neighbors_minHash = 10, size_of_query = 50, number_of_hashfunctions=[400]):
    """Function to measure and plot the performance for the given input data.
        For the query times two methods are measured:
            - All queries at once
            - One query after another
        For example:
            - For 50 queries kneighbors is called once with 50 instances to get all neighbors of these instances
            - For 50 queries kneighbors is called 50 times with only one instance."""

    time_fit_sklearn = []
    time_fit_minHash = []
    time_fit_lshf = []
    time_fit_annoy = []
    time_fit_flann = []

    time_query_time_50_1_sklearn = []
    time_query_time_50_1_minHash_exact = []
    time_query_time_50_1_minHash_approx = []
    time_query_time_50_1_lshf = []
    time_query_time_50_1_annoy = []
    time_query_time_50_1_flann = []

    time_query_time_1_50_sklearn = []
    time_query_time_1_50_minHash_exact = []
    time_query_time_1_50_minHash_approx = []
    time_query_time_1_50_lshf = []
    time_query_time_1_50_annoy = []
    time_query_time_1_50_flann = []

    accuracy_1_50_lshf = []
    accuracy_1_50_minHash_exact = []
    accuracy_1_50_minHash_aprox = []
    accuracy_1_50_annoy = []
    accuracy_1_50_flann = []

    centroids = 8
    size_of_datasets = 7
    # length_of_dataset = len(dataset)
    iteration_of_dataset = 0

    for hash_function in number_of_hashfunctions:
        iteration_of_dataset += 1
        # print "Dataset processing: ", iteration_of_dataset, "/", length_of_dataset
        nearest_neighbor_sklearn = NearestNeighbors(n_neighbors = n_neighbors_sklearn)
        nearest_neighbor_minHash = MinHash(n_neighbors = n_neighbors_minHash, number_of_hash_functions=hash_function,
                         max_bin_size= 36, shingle_size = 4, similarity=False, 
                        number_of_cores=4,
                     prune_inverse_index=2, store_value_with_least_sigificant_bit=1, excess_factor=11,
                    prune_inverse_index_after_instance=0.5, remove_hash_function_with_less_entries_as=4, 
                     shingle=1, block_size=2)
        nearest_neighbor_lshf = LSHForest(n_estimators=20, n_candidates=200, n_neighbors=n_neighbors_minHash)
        time_start = time.time()
        nearest_neighbor_sklearn.fit(dataset)
        time_end = time.time()
        time_fit_sklearn.append(time_end - time_start)
        # print "Fitting of sklearn_nneighbors done!"
        time_start = time.time()
        nearest_neighbor_minHash.fit(dataset)
        time_end = time.time()
        time_fit_minHash.append(time_end - time_start)
        # print "Fitting of minHash_nneighbors done!"
        data_projection = SparseRandomProjection(n_components=hash_function, random_state=1)
        dataset_dense = data_projection.fit_transform(dataset)
        
        time_start = time.time()
        nearest_neighbor_lshf.fit(dataset_dense)
        time_end = time.time()
        time_fit_lshf.append(time_end - time_start)
        # print "Fitting of LSHF done!"

        time_start = time.time()
        annoy_ = annoy.AnnoyIndex(f=dataset_dense.shape[1])
        for i, x in enumerate(dataset_dense):
            annoy_.add_item(i, x.toarray()[0])
        annoy_.build(100) # ntrees = 100
        time_end = time.time()
        time_fit_annoy.append(time_end - time_start)

        if size_of_query is None:
            size_of_query = dataset.shape[0]
        if size_of_query < dataset.shape[0]:
            query_ids = set()
            query_list = []
            for i in range(50):
                query_ids.add(random.randint(0, dataset.shape[0]-1))
            query_dense = dataset_dense[list(query_ids)]   
            query_dense_annoy = dataset_dense[list(query_ids)]
            for i in query_ids:
                query_list.append(dataset.getrow(i))
            query = vstack(query_list)
        else:
            query = None
            query_dense = None
            query_dense_annoy = dataset_dense
        time_start = time.time()
        n_neighbors_sklearn_1_50 = nearest_neighbor_sklearn.kneighbors(query, return_distance=False)
        time_end = time.time()
        time_query_time_1_50_sklearn.append(time_end - time_start)
        time_start = time.time()
        n_neighbors_minHash_exact_1_50 = nearest_neighbor_minHash.kneighbors(query, return_distance=False)
        time_end = time.time()
        time_query_time_1_50_minHash_exact.append(time_end - time_start)

        time_start = time.time()
        n_neighbors_minHash_approx_1_50 = nearest_neighbor_minHash.kneighbors(query, fast=True, return_distance=False)
        time_end = time.time()
        time_query_time_1_50_minHash_approx.append(time_end - time_start)
        time_start = time.time()
        n_neighbors_lshf_1_50 = nearest_neighbor_lshf.kneighbors(query_dense_annoy,return_distance=False)
        time_end = time.time()
        time_query_time_1_50_lshf.append(time_end - time_start)

        # accuracy score position wise for lshf
        accuracy_score_ = 0.0
        for x, y in zip(n_neighbors_lshf_1_50, n_neighbors_sklearn_1_50):
            accuracy_score_ += accuracy_score(x, y)
        accuracy_score_ = accuracy_score_ / float(len(n_neighbors_lshf_1_50))
        accuracy_1_50_lshf.append(accuracy_score_)
        
        # accuracy score for minHash fast
        accuracy_score_ = 0.0
        for x, y in zip(n_neighbors_minHash_approx_1_50, n_neighbors_sklearn_1_50):
            accuracy_score_ += accuracy_score(x, y)
        accuracy_score_ = accuracy_score_ / float(len(n_neighbors_minHash_approx_1_50))
        accuracy_1_50_minHash_aprox.append(accuracy_score_)
         
        # accuracy score for minHash non-fast
        accuracy_score_ = 0.0
        for x, y in zip(n_neighbors_minHash_exact_1_50, n_neighbors_sklearn_1_50):
            accuracy_score_ += accuracy_score(x, y)
        accuracy_score_ = accuracy_score_ / float(len(n_neighbors_minHash_exact_1_50))
        accuracy_1_50_minHash_exact.append(accuracy_score_)
        
       
        if size_of_query >= dataset.shape[0]:
            query = dataset
            query_dense = dataset_dense
            query_dense_annoy = dataset_dense
            
        time_query_time_50_1_sklearn_loc = []
        time_query_time_50_1_sklearn_loc = []
        for i in xrange(size_of_query-1):
            time_start = time.time()
            nearest_neighbor_sklearn.kneighbors(query[i],return_distance=False)
            time_end = time.time()
            time_query_time_50_1_sklearn_loc.append(time_end - time_start)
        time_query_time_50_1_sklearn.append(np.sum(time_query_time_50_1_sklearn_loc))

        time_query_time_50_1_minHash_exact_loc = []
        for i in xrange(size_of_query-1):
            time_start = time.time()
            nearest_neighbor_minHash.kneighbors(query[i], return_distance=False)
            time_end = time.time()
            time_query_time_50_1_minHash_exact_loc.append(time_end - time_start)
        time_query_time_50_1_minHash_exact.append(np.sum(time_query_time_50_1_minHash_exact_loc))

        time_query_time_50_1_minHash_approx_loc = []
        for i in xrange(size_of_query-1):
            time_start = time.time()
            nearest_neighbor_minHash.kneighbors(query[i], fast=True,return_distance=False)
            time_end = time.time()
            time_query_time_50_1_minHash_approx_loc.append(time_end - time_start)
        time_query_time_50_1_minHash_approx.append(np.sum(time_query_time_50_1_minHash_approx_loc))

        time_query_time_50_1_lshf_loc = []
        for i in xrange(size_of_query-1):
            time_start = time.time()
            nearest_neighbor_lshf.kneighbors(query_dense[i], return_distance=False)
            time_end = time.time()
            time_query_time_50_1_lshf_loc.append(time_end - time_start)
        time_query_time_50_1_lshf.append(np.sum(time_query_time_50_1_lshf_loc))

        time_query_time_50_1_annoy_loc = []
        n_neighbors_annoy_1_50 = []
        for i in xrange(size_of_query-1):
            time_start = time.time()
            nearest_neighbor_annoy = annoy_.get_nns_by_vector(query_dense_annoy[i].toarray()[0], n_neighbors_sklearn, 100)
            time_end = time.time()
            time_query_time_50_1_annoy_loc.append(time_end - time_start)
            n_neighbors_annoy_1_50.append(nearest_neighbor_annoy)
        time_query_time_50_1_annoy.append(np.sum(time_query_time_50_1_annoy_loc))
        
        # accuracy score for annoy
        accuracy_score_ = 0.0
        for x, y in zip(n_neighbors_annoy_1_50, n_neighbors_sklearn_1_50):
            accuracy_score_ += accuracy_score(x, y)
        accuracy_score_ = accuracy_score_ / float(len(n_neighbors_annoy_1_50))
        accuracy_1_50_annoy.append(accuracy_score_)


    return  (time_fit_sklearn, 
                time_fit_minHash, 
                time_fit_lshf,
                time_fit_annoy,
            time_query_time_50_1_sklearn,
            time_query_time_50_1_minHash_exact, 
            time_query_time_50_1_minHash_approx, 
            time_query_time_50_1_lshf, 
            time_query_time_50_1_annoy,
            time_query_time_1_50_sklearn,
            time_query_time_1_50_minHash_exact, 
            time_query_time_1_50_minHash_approx,
            time_query_time_1_50_lshf, 
            accuracy_1_50_lshf,
            accuracy_1_50_minHash_exact, 
            accuracy_1_50_minHash_aprox, 
            accuracy_1_50_annoy)