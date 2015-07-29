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
        algorithm : {'approximate', 'exact'}, optional (default = 'approximate')
            - 'approximate':    will only use an inverse index to compute a k_neighbor query.
            - 'exact':          an inverse index is used to preselect instances, and these are used to get
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

    def __init__(self, n_neighbors=5, algorithm="approximate", number_of_hash_functions=400,
                 max_bin_size = 50, minimal_blocks_in_common = 1, block_size = 4, excess_factor = 5, number_of_cores=None,
                 chunk_size=None):
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.nearestNeighbors = MinHashNearestNeighbors(n_neighbors=n_neighbors, algorithm = algorithm,
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

    def kneighbors(self, X = None, n_neighbors = None, return_distance = True, algorithm=None):
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
            algorithm : {'exact', 'approximate'}
                - 'approximate':    will only use an inverse index to compute a k_neighbor query.
                - 'exact':          an inverse index is used to preselect instances, and these are used to get
                                    the original data from the data set to answer a k_neighbor query. The
                                    original data is stored in the memory.
                If not defined, default value given by constructor is used.
            Returns
            -------
            dist : array
                Array representing the lengths to points, only present if
                return_distance=True
            ind : array
                Indices of the nearest points in the population matrix."""
        if algorithm is not None:
            self.algorithm = algorithm
        return self.nearestNeighbors.kneighbors(X=X, n_neighbors=n_neighbors, return_distance=return_distance, algorithm=self.algorithm)


    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity', algorithm=None):
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
            algorithm : {'exact', 'approximate'}
                - 'approximate':    will only use an inverse index to compute a k_neighbor query.
                - 'exact':          an inverse index is used to preselect instances, and these are used to get
                                    the original data from the data set to answer a k_neighbor query. The
                                    original data is stored in the memory.
                If not defined, default value given by constructor is used.
            Returns
            -------
            A : sparse matrix in CSR format, shape = [n_samples, n_samples_fit]
                n_samples_fit is the number of samples in the fitted data
                A[i, j] is assigned the weight of edge that connects i to j."""
        if algorithm is not None:
            self.algorithm = algorithm
        return self.nearestNeighbors.kneighbors_graph(X=X, n_neighbors=n_neighbors, mode=mode, algorithm=self.algorithm)


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
        predicted_class_list = []
        X = check_array(X, accept_sparse='csr')
        number_of_instances = X.shape[0]
        n_neighbors = self.n_neighbors
        for i in xrange(number_of_instances):
            # compute signature for one instance
            signature = self.nearestNeighbors._inverseIndex.signature(X[i])
            # get candidates from inverse index
            distance, candidate_list = self.nearestNeighbors._inverseIndex.neighbors(signature, self.n_neighbors)
            # compute the prediction with the KNeighborsClassifier from sklearn
            # use original data in case of "exact" otherwise use signatures.
            n_neighbors = self.n_neighbors
            if len(candidate_list[0]) < self.n_neighbors:
                n_neighbors = len(candidate_list[0])
            if self.algorithm == "exact":
                nearest_neighbors = KNeighborsClassifier(n_neighbors=n_neighbors)
                label_list = []
                for j in candidate_list[0]:
                    label_list.append(self.nearestNeighbors._y[j])
                nearest_neighbors.fit(
                    self.nearestNeighbors._X[[candidate_list[0]]], label_list)
                predicted_class_list.extend(nearest_neighbors.predict(X[i]))
            else:
                nearest_neighbors = KNeighborsClassifier(n_neighbors=n_neighbors)
                label_list = []
                for j in candidate_list[0]:
                    label_list.append(self.nearestNeighbors._y[j])
                nearest_neighbors.fit(
                    self.nearestNeighbors._inverseIndex.get_signature_list(self.nearestNeighbors._X[[candidate_list[0]]]),
                    label_list)
                predicted_class_list.extend(nearest_neighbors.predict(signature))
        return predicted_class_list

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
        predicted_class_list = []
        X = check_array(X, accept_sparse='csr')
        number_of_instances = X.shape[0]

        for i in xrange(number_of_instances):
            # compute signature for instance
            signature = self.nearestNeighbors._inverseIndex.signature(X[i])
            # get all neighbors from inverse index
            distance, candidate_list = self.nearestNeighbors._inverseIndex.neighbors(signature, self.n_neighbors)
            # compute the prediction with the KNeighborsClassifier from sklearn
            # use original data in case of "exact" otherwise use signatures.
            n_neighbors = self.n_neighbors
            if len(candidate_list[0]) < self.n_neighbors:
                n_neighbors = len(candidate_list[0])
            if self.algorithm == "exact":
                nearest_neighbors = KNeighborsClassifier(n_neighbors=n_neighbors)
                label_list = []
                for j in candidate_list[0]:
                    label_list.append(self.nearestNeighbors._y[j])
                nearest_neighbors.fit(self.nearestNeighbors._X[[candidate_list[0]]], label_list)
                predicted_class_list.extend(nearest_neighbors.predict_proba(X[i]))
            else:
                nearest_neighbors = KNeighborsClassifier(n_neighbors=n_neighbors)
                label_list = []
                for j in candidate_list[0]:
                    label_list.append(self.nearestNeighbors._y[j])
                nearest_neighbors.fit(self.nearestNeighbors._inverseIndex.get_signature_list(self.nearestNeighbors._X[[candidate_list[0]]])
                                      , label_list)
                predicted_class_list.extend(nearest_neighbors.predict_proba(signature))
        return predicted_class_list
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
        score_list = []
        X = check_array(X, accept_sparse='csr')
        number_of_instances = X.shape[0]

        for i in xrange(number_of_instances):
            # compute signature for instance
            signature = self.nearestNeighbors._inverseIndex.signature(X[i])
            # get all neighbors from inverse index
            distance, candidate_list = self.nearestNeighbors._inverseIndex.neighbors(signature, self.n_neighbors)
            # compute the score with the KNeighborsClassifier from sklearn
            # use original data in case of "exact" otherwise use signatures.
            n_neighbors = self.n_neighbors
            if len(candidate_list[0]) < self.n_neighbors:
                n_neighbors = len(candidate_list[0])
            if self.algorithm == "exact":
                nearest_neighbors = KNeighborsClassifier(n_neighbors=n_neighbors)
                label_list = []
                for j in candidate_list[0]:
                    label_list.append(self.nearestNeighbors._y[j])
                nearest_neighbors.fit(self.nearestNeighbors._X[candidate_list[0]], label_list)
                score_list.append(nearest_neighbors.score(X[i], y, sample_weight))
            else:
                nearest_neighbors = KNeighborsClassifier(n_neighbors=n_neighbors)
                label_list = []
                for j in candidate_list[0]:
                    label_list.append(self.nearestNeighbors._y[j])
                nearest_neighbors.fit(
                    self.nearestNeighbors._inverseIndex.get_signature_list(self.nearestNeighbors._X[candidate_list[0]]),
                    label_list)
                score_list.append(nearest_neighbors.score(signature, y, sample_weight))
        return score_list
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        pass
