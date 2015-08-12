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

from scipy.sparse import dok_matrix
from scipy.sparse import rand
from scipy.sparse import vstack

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
