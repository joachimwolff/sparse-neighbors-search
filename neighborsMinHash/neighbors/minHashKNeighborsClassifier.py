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

from collections import Counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_array
import logging

from . import MinHashNearestNeighbors

logger = logging.getLogger(__name__)
class MinHashKNeighborsClassifier():
    """Classifier implementing the k-nearest neighbors vote on sparse data sets.
        Based on a dimension reduction with minimum hash functions.
        
        Parameters
        ----------
        n_neighbors : int, optional (default = 5)
            Number of neighbors to use by default for :meth:`k_neighbors` queries.
        fast : {True, False}, optional (default = False)
            - True:     will only use an inverse index to compute a k_neighbor query.
            - False:    an inverse index is used to preselect instances, and these are used to get
                        the original data from the data set to answer a k_neighbor query. The
                        original data is stored in the memory.
        number_of_hash_functions : int, optional (default = '400')
            Number of hash functions to use for computing the inverse index.
        max_bin_size : int, optional (default = 50)
            The number of maximal collisions for one hash value of one hash function. If one value of a hash function
            has more collisions, this value will be ignored.
        minimal_blocks_in_common : int, optional (default = 1)
            The minimal number of hash collisions two instances have to be in common to be recognised. Everything less
            will be ignored.
        block_size : int, optional (default = 4)
            Reduction factor for the signature size.
            E.g. number_of_hash_functions=400 and block_size=4 --> Size of the signature will be 100
        excess_factor : int, optional (default = 5)
            Factor to return more neighbors internally as defined with n_neighbors. Factor is useful to increase the
            precision of the :meth:`algorithm=exact` version of the implementation.
            E.g.: n_neighbors = 5, excess_factor = 5. Internally n_neighbors*excess_factor = 25 neighbors will be returned.
            Now the reduced data set for sklearn.NearestNeighbors is of size 25 and not 5.
        number_of_cores : int, optional
            Number of cores that should be used for openmp. If your system doesn't support openmp, this value
            will have no effect. If it supports openmp and it is not defined, the maximum number of cores is used.
        chunk_size : int, optional
            Number of elements one cpu core should work on. If it is set to "0" the default behaviour of openmp is used;
            e.g. for an 8-core cpu,  the chunk_size is set to 8. Every core will get 8 elements, process these and get
            another 8 elements until everything is done. If you set chunk_size to "-1" all cores
            are getting the same amount of data at once; e.g. 8-core cpu and 128 elements to process, every core will
            get 16 elements at once.
        
        Notes
        -----

        The documentation is copied from scikit-learn and was only extend for a few cases. All examples are available there.
        Original documentation is available at: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
        
        Sources:
        Basic algorithm:
        http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

        Idea behind implementation:
        https://en.wikipedia.org/wiki/Locality-sensitive_hashing

        Implementation is using scikit learn:
        http://scikit-learn.org/dev/index.html
        http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

        Algorithm based on:
        Heyne, S., Costa, F., Rose, D., & Backofen, R. (2012).
        GraphClust: alignment-free structural clustering of local RNA secondary structures.
        Bioinformatics, 28(12), i224-i232.
        http://bioinformatics.oxfordjournals.org/content/28/12/i224.full.pdf+html"""

    def __init__(self, n_neighbors=5, fast=False, number_of_hash_functions=400,
                 max_bin_size = 50, minimal_blocks_in_common = 1, block_size = 4, excess_factor = 5, number_of_cores=None,
                 chunk_size=None):
        self.n_neighbors = n_neighbors
        if fast is not None:
            self.fast = fast
        else:
            self.fast = False
        self.nearestNeighbors = MinHashNearestNeighbors(n_neighbors=n_neighbors, fast = fast,
                                                        number_of_hash_functions=number_of_hash_functions,
                                                        max_bin_size =max_bin_size,
                                                        minimal_blocks_in_common=minimal_blocks_in_common,
                                                        block_size=block_size,
                                                        excess_factor=excess_factor,
                                                        number_of_cores=number_of_cores,
                                                        chunk_size=chunk_size)
    def fit(self, X, y):
        """Fit the model using X as training data.

            Parameters
            ----------
            X : {array-like, sparse matrix}
                Training data, shape = [n_samples, n_features]
            y : {array-like, sparse matrix}
                Target values of shape = [n_samples] or [n_samples, n_outputs]"""
        self.nearestNeighbors.fit(X, y)
    def partial_fit(self, X, y):
        """Extend the model by X as additional training data.

            Parameters
            ----------
            X : {array-like, sparse matrix}
                Training data. Shape = [n_samples, n_features]
            y : {array-like, sparse matrix}
                Target values of shape = [n_samples] or [n_samples, n_outputs]"""
        self.nearestNeighbors.partial_fit(X, y)
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        print "n_neighbors: ", self.n_neighbors
        print "algorithm: ", self.algorithm

    def kneighbors(self, X = None, n_neighbors = None, return_distance = True, fast=None):
        """Finds the K-neighbors of a point.

            Returns distance

            Parameters
            ----------
            X : array-like, last dimension same as that of fit data, optional
                The query point or points.
                If not provided, neighbors of each indexed point are returned.
                In this case, the query point is not considered its own neighbor.
            n_neighbors : int
                Number of neighbors to get (default is the value
                passed to the constructor).
            return_distance : boolean, optional. Defaults to True.
                If False, distances will not be returned
            fast : {True, False}, optional (default = False)
                - True:     will only use an inverse index to compute a k_neighbor query.
                - False:    an inverse index is used to preselect instances, and these are used to get
                            the original data from the data set to answer a k_neighbor query. The
                            original data is stored in the memory.
                If not defined, default value given by constructor is used.
            Returns
            -------
            dist : array, shape = [n_samples, distance]
                Array representing the lengths to points, only present if
                return_distance=True
            ind : array, shape = [n_samples, neighbors]
                Indices of the nearest points in the population matrix."""
        
        if fast is not None:
            self.fast = fast

        return self.nearestNeighbors.kneighbors(X=X, n_neighbors=n_neighbors, return_distance=return_distance, fast=fast)


    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity', fast=None):
        """Computes the (weighted) graph of k-Neighbors for points in X
            Parameters
            ----------
            X : array-like, last dimension same as that of fit data, optional
                The query point or points.
                If not provided, neighbors of each indexed point are returned.
                In this case, the query point is not considered its own neighbor.
            n_neighbors : int
                Number of neighbors for each sample.
                (default is value passed to the constructor).
            mode : {'connectivity', 'distance'}, optional
                Type of returned matrix: 'connectivity' will return the
                connectivity matrix with ones and zeros, in 'distance' the
                edges are Euclidean distance between points.
            fast : {True, False}, optional (default = False)
                - True:     will only use an inverse index to compute a k_neighbor query.
                - False:    an inverse index is used to preselect instances, and these are used to get
                            the original data from the data set to answer a k_neighbor query. The
                            original data is stored in the memory.
                If not defined, default value given by constructor is used.
            Returns
            -------
            A : sparse matrix in CSR format, shape = [n_samples, n_samples_fit]
                n_samples_fit is the number of samples in the fitted data
                A[i, j] is assigned the weight of edge that connects i to j."""
        if fast is not None:
            self.fast = fast
        return self.nearestNeighbors.kneighbors_graph(X=X, n_neighbors=n_neighbors, mode=mode, fast=fast)


    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
            X : array of shape [n_samples, n_features]
                A 2-D array representing the test points.
            Returns
            -------
            y : array of shape [n_samples] or [n_samples, n_outputs]
                Class labels for each data sample.
        """
        X = check_array(X, accept_sparse='csr')
        n_neighbors = self.n_neighbors
        inverseIndex = self.nearestNeighbors._inverseIndex

        # compute neighbors in inverse index
        _, neighbors = inverseIndex.neighbors(X, n_neighbors)
        candidate_list = []
        for i in xrange(len(neighbors)):
           candidate_list.extend(neighbors[i])
        candidate_set = set(candidate_list)
        candidate_list = list(candidate_set)

        if len(candidate_list) < n_neighbors:
            n_neighbors = len(candidate_list)

        # signatures = inverseIndex.signature(X)
        distance, neighbors = self.kneighbors(X)
        y_values = self._getYValues(candidate_list)

        # print neighbors
        result_classification = []
        for instance in neighbors:
            y_value = []
            for instance_ in instance:
                if instance_ != -1:
                # get all class labels
                    y_value.append(y_values[instance_])
            if len(y_value) > 0:
                # sort class labels by frequency and take the highest one
                result_classification.append(Counter(y_value).keys()[0])
            else:
                result_classification.append(-1)
        return result_classification



    def predict_proba(self, X):
        """Return probability estimates for the test data X.
            Parameters
            ----------
            X : array, shape = (n_samples, n_features)
                A 2-D array representing the test points.
            Returns
            -------
            p : array of shape = [n_samples, n_classes], or a list of n_outputs
                of such arrays if n_outputs > 1.
                The class probabilities of the input samples. Classes are ordered
                by lexicographic order.
        """
        X = check_array(X, accept_sparse='csr')
        n_neighbors = self.n_neighbors
        inverseIndex = self.nearestNeighbors._inverseIndex
        number_of_classes = len(set(self.nearestNeighbors._y))

        # compute neighbors in inverse index
        _, neighbors = inverseIndex.neighbors(X, n_neighbors)
        candidate_list = []
        for i in xrange(len(neighbors)):
           candidate_list.extend(neighbors[i])
        candidate_set = set(candidate_list)
        candidate_list = list(candidate_set)

        if len(candidate_list) < n_neighbors:
            n_neighbors = len(candidate_list)

        distance, neighbors = self.kneighbors(X)
        y_values = self._getYValues(candidate_list)

        result_classification = []
        for instance in neighbors:
            y_value = []
            for instance_ in instance:
                if instance_ != -1:
                # get all class labels
                    y_value.append(y_values[instance_])
            if len(y_value) > 0:
                # sort class labels by frequency
                y_proba = [0.0] * number_of_classes
                sorted_classes = Counter(y_value)
                # count frequency of all clases
                total_class_count = 0
                for value in sorted_classes.itervalues():
                    total_class_count += value
                # compute probability by frequency / all_frequencies
                for key, value in sorted_classes.iteritems():
                    y_proba[key] = value / float(total_class_count)
                result_classification.append(y_proba[:])
        return result_classification
        
    def score(self, X, y , sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        X = check_array(X, accept_sparse='csr')
        n_neighbors = self.n_neighbors
        inverseIndex = self.nearestNeighbors._inverseIndex

         # compute neighbors in inverse index
        _, neighbors = inverseIndex.neighbors(X, n_neighbors)
        candidate_list = []
        for i in xrange(len(neighbors)):
           candidate_list.extend(neighbors[i])
        candidate_set = set(candidate_list)
        candidate_list = list(candidate_set)

        if len(candidate_list) < n_neighbors:
            n_neighbors = len(candidate_list)

        nearest_neighborsSK = KNeighborsClassifier(n_neighbors=n_neighbors)

        # if fast:
        #     signatures = inverseIndex.signature(X)
        #     nearest_neighborsSK.fit(
        #         inverseIndex.get_signature_list(self.nearestNeighbors._X[candidate_list]),
        #         self._getYValues(candidate_list))
        #     return nearest_neighborsSK.score(signatures, y, sample_weight)
        # else:
        nearest_neighborsSK.fit(
            self.nearestNeighbors._X[candidate_list], self._getYValues(candidate_list))
        return nearest_neighborsSK.score(X, y, sample_weight)

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        pass

    def _getYValues(self, candidate_list):
        if self.nearestNeighbors._y_is_csr:
            return self.nearestNeighbors._y[candidate_list]
        else:
            y_list = []
            for i in xrange(len(candidate_list)):
                y_list.append(self.nearestNeighbors._y[candidate_list[i]])
            return y_list
