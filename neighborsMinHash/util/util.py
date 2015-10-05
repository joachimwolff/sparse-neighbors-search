# Copyright 2015 Joachim Wolff
# Master Project
# Tutors: Milad Miladi, Fabrizio Costa
# Summer semester 2015
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwig-University Freiburg im Breisgau

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
import sklearn

from ..neighbors import MinHashNearestNeighbors
import pyflann
import annoy

def accuracy(neighbors_exact, neighbors_approx, neighbors_sklearn):
    """Computes the accuracy for the exact and approximate version of the minHashNearestNeighbors algorithm.

    Parameters
    ----------
    neighbors_exact : array[[neighbors]]
    neighbors_approx : array[[neighbors]]
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
    matches = 0
    all_elements = 0
    for i in range(len(neighbors_exact)):
        for j in neighbors_exact[i]:
            if j in neighbors_sklearn[i]:
                matches += 1
        all_elements += len(neighbors_exact[i])
    # exact vs sklearn
    exact_sklearn = matches / float(all_elements) 

    matches = 0
    all_elements = 0
    for i in range(len(neighbors_approx)):
        for j in neighbors_approx[i]:
            if j in neighbors_sklearn[i]:
                matches += 1
        all_elements += len(neighbors_exact[i])
    # approx vs. sklearn
    approx_sklearn = matches / float(all_elements) 

    matches = 0
    all_elements = 0
    for i in range(len(neighbors_approx)):
        for j in neighbors_approx[i]:
            if j in neighbors_exact[i]:
                matches += 1
        all_elements += len(neighbors_exact[i])
    # approx vs. exact
    approx_exact = matches / float(all_elements) 
    return exact_sklearn, approx_sklearn, approx_exact

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

def measure_performance(dataset, n_neighbors_sklearn = 5, n_neighbors_minHash = 5, size_of_query = 50, number_of_hashfunctions=400):
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
    length_of_dataset = len(dataset)
    iteration_of_dataset = 0
    for dataset_ in dataset:
        iteration_of_dataset += 1
        print "Dataset processing: ", iteration_of_dataset, "/", length_of_dataset
        nearest_neighbor_sklearn = NearestNeighbors(n_neighbors = n_neighbors_sklearn)
        nearest_neighbor_minHash = MinHashNearestNeighbors(n_neighbors = n_neighbors_minHash, number_of_hash_functions=number_of_hashfunctions)
        nearest_neighbor_lshf = LSHForest(n_estimators=20, n_candidates=200, n_neighbors=n_neighbors_minHash)
        time_start = time.time()
        nearest_neighbor_sklearn.fit(dataset_)
        time_end = time.time()
        time_fit_sklearn.append(time_end - time_start)
        print "Fitting of sklearn_nneighbors done!"
        time_start = time.time()
        nearest_neighbor_minHash.fit(dataset_)
        time_end = time.time()
        time_fit_minHash.append(time_end - time_start)
        print "Fitting of minHash_nneighbors done!"

        time_start = time.time()
        nearest_neighbor_lshf.fit(dataset_)
        time_end = time.time()
        time_fit_lshf.append(time_end - time_start)
        print "Fitting of LSHF done!"

        time_start = time.time()
        annoy_ = annoy.AnnoyIndex(f=dataset_.shape[1])
        for i, x in enumerate(dataset_):
            annoy_.add_item(i, x.toarray()[0])
        annoy_.build(100) # ntrees = 100
        time_end = time.time()
        time_fit_annoy.append(time_end - time_start)
        print "Fitting of annoy done!"

        # time_start = time.time()
        # flann_ = pyflann.FLANN(target_precision=0.95, algorithm='autotuned', log_level='info')
        # X = sklearn.preprocessing.normalize(dataset_, axis=1, norm='l2')
        # # print "X.shape: ", X.shape
        # # print "dataset_.shape: ", dataset_.shape
        # flann_.build_index(X.todense())
        # time_end = time.time()
        # time_fit_flann.append(time_end - time_start)
        # print "Fitting of flann done!"


        if size_of_query < dataset_.shape[0]:
            query_ids = []
            for i in range(size_of_query):
                query_ids.append(random.randint(0, centroids * size_of_datasets))
            query = dataset_[query_ids]
        else:
            query = dataset_

        time_start = time.time()
        n_neighbors_sklearn_1_50 = nearest_neighbor_sklearn.kneighbors(query, return_distance=False)
        time_end = time.time()
        time_query_time_1_50_sklearn.append(time_end - time_start)
        print "Computation of sklearn_nneighbors done!"

        time_start = time.time()
        n_neighbors_minHash_exact_1_50 = nearest_neighbor_minHash.kneighbors(query, return_distance=False)
        time_end = time.time()
        time_query_time_1_50_minHash_exact.append(time_end - time_start)
        print "Computation of minHash_slow done!"

        time_start = time.time()
        n_neighbors_minHash_approx_1_50 = nearest_neighbor_minHash.kneighbors(query, fast=True, return_distance=False)
        time_end = time.time()
        time_query_time_1_50_minHash_approx.append(time_end - time_start)
        print "Computation of minHash_fast done!"

        time_start = time.time()
        n_neighbors_lshf_1_50 = nearest_neighbor_lshf.kneighbors(query,return_distance=False)
        time_end = time.time()
        time_query_time_1_50_lshf.append(time_end - time_start)
        print "Computation of lshf done!"


        # time_start = time.time()
        # n_neighbors_annoy_1_50 = annoy_.get_nns_by_vector(query.toarray()[0], n_neighbors_sklearn, 100)
        # print "n_neighbors_annoy_1_50: ", n_neighbors_annoy_1_50
        # time_end = time.time()
        # time_query_time_1_50_annoy.append(time_end - time_start)

        # print "Computation of annoy done!"

        # time_start = time.time()
        # v = sklearn.preprocessing.normalize(query, axis=1, norm='l2')[0]
        # n_neighbors_flann_1_50 = flann_.nn_index(v, None)[0][0]
        # print "n_neighbors_flann_1_50: ", n_neighbors_flann_1_50
        # print "n_neighbors_sklearn_1_50: ", n_neighbors_sklearn_1_50
        # time_end = time.time()
        # time_query_time_1_50_flann.append(time_end - time_start)

        # print "Computation of flann done!"

        accuracy_1_50_lshf.append(np.in1d(n_neighbors_lshf_1_50, n_neighbors_sklearn_1_50).mean())
        accuracy_1_50_minHash_exact.append(np.in1d(n_neighbors_minHash_exact_1_50, n_neighbors_sklearn_1_50).mean())
        accuracy_1_50_minHash_aprox.append(np.in1d(n_neighbors_minHash_approx_1_50, n_neighbors_sklearn_1_50).mean())
        # accuracy_1_50_annoy.append(np.in1d(n_neighbors_annoy_1_50, n_neighbors_sklearn_1_50).mean())
        # accuracy_1_50_flann.append(np.in1d(n_neighbors_flann_1_50, n_neighbors_sklearn_1_50).mean())

        time_query_time_50_1_sklearn_loc = []
        time_query_time_50_1_sklearn_loc = []
        for i in range(size_of_query):
            time_start = time.time()
            nearest_neighbor_sklearn.kneighbors(query[i],return_distance=False)
            time_end = time.time()
            time_query_time_50_1_sklearn_loc.append(time_end - time_start)
        time_query_time_50_1_sklearn.append(np.sum(time_query_time_50_1_sklearn_loc))
        print "Computation_2 of sklearn_nneighbors done!"

        time_query_time_50_1_minHash_exact_loc = []
        for i in range(size_of_query):
            time_start = time.time()
            nearest_neighbor_minHash.kneighbors(query[i], return_distance=False)
            time_end = time.time()
            time_query_time_50_1_minHash_exact_loc.append(time_end - time_start)
        time_query_time_50_1_minHash_exact.append(np.sum(time_query_time_50_1_minHash_exact_loc))
        print "Computation_2 of minHash_slow done!"

        time_query_time_50_1_minHash_approx_loc = []
        for i in range(size_of_query):
            time_start = time.time()
            nearest_neighbor_minHash.kneighbors(query[i], fast=True,return_distance=False)
            time_end = time.time()
            time_query_time_50_1_minHash_approx_loc.append(time_end - time_start)
        time_query_time_50_1_minHash_approx.append(np.sum(time_query_time_50_1_minHash_approx_loc))
        print "Computation_2 of minHash_fast done!"

        time_query_time_50_1_lshf_loc = []
        for i in range(size_of_query):
            time_start = time.time()
            nearest_neighbor_lshf.kneighbors(query[i], return_distance=False)
            time_end = time.time()
            time_query_time_50_1_lshf_loc.append(time_end - time_start)
        time_query_time_50_1_lshf.append(np.sum(time_query_time_50_1_lshf_loc))
        print "Computation_2 of lshf done!"

        time_query_time_50_1_annoy_loc = []
        n_neighbors_annoy_1_50 = []
        for i in range(size_of_query):
            time_start = time.time()
            # print "query[i].toarray(): ", query[i].toarray()
            nearest_neighbor_annoy = annoy_.get_nns_by_vector(query[i].toarray()[0], n_neighbors_sklearn, 100)
            time_end = time.time()
            time_query_time_50_1_annoy_loc.append(time_end - time_start)
            n_neighbors_annoy_1_50.append(nearest_neighbor_annoy)
        time_query_time_50_1_annoy.append(np.sum(time_query_time_50_1_annoy_loc))
        accuracy_1_50_annoy.append(np.in1d(n_neighbors_annoy_1_50, n_neighbors_sklearn_1_50).mean())

        print "Computation_2 of annoy done!"

        # time_query_time_50_1_flann_loc = []
        # n_neighbors_flann_1_50 = []
        # for i in range(size_of_query):
        #     time_start = time.time()
        #     v = sklearn.preprocessing.normalize(query[i].toarray()[0], axis=1, norm='l2')[0]
        #     nearest_neighbor_flann = flann_.nn_index(v, n_neighbors_sklearn)[0][0]
        #     n_neighbors_flann_1_50.append(nearest_neighbor_flann)
        #     time_end = time.time()
        #     time_query_time_50_1_flann_loc.append(time_end - time_start)
        # time_query_time_50_1_flann.append(np.sum(time_query_time_50_1_flann_loc))
        # accuracy_1_50_flann.append(np.in1d(n_neighbors_flann_1_50, n_neighbors_sklearn_1_50).mean())

        # print "Computation_2 of flann done!"


    return  (time_fit_sklearn, time_fit_minHash, time_fit_lshf, time_fit_annoy,  
            time_query_time_50_1_sklearn,time_query_time_50_1_minHash_exact, time_query_time_50_1_minHash_approx, 
            time_query_time_50_1_lshf,
            time_query_time_1_50_sklearn,time_query_time_1_50_minHash_exact, time_query_time_1_50_minHash_approx,
            time_query_time_1_50_lshf, time_query_time_1_50_annoy,
            accuracy_1_50_lshf,accuracy_1_50_minHash_exact, accuracy_1_50_minHash_aprox, accuracy_1_50_annoy)
