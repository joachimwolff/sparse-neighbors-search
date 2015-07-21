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
import logging

import numpy as np
import multiprocessing as mp
from math import ceil
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

from computation import InverseIndex
# from . import apply_async


logger = logging.getLogger(__name__)

class MinHashNearestNeighbors():
    """Approximate unsupervised learner for implementing neighbor searches on sparse data sets. Based on a
        dimension reduction with minimum hash functions.

        Parameters
        ----------
        n_neighbors : int, optional (default = 5)
            Number of neighbors to use by default for :meth:`k_neighbors` queries.
        radius : float, optional (default = 1.0)
            Range of parameter space to use by default for :meth`radius_neighbors`
            queries.

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
        
        Notes
        -----

        The documentation is copied from scikit-learn and was only extend for a few cases. All examples are available there.
        http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors

        Sources:
        Basic algorithm:
        http://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm

        Idea behind implementation:
        https://en.wikipedia.org/wiki/Locality-sensitive_hashing

        Implementation is using scikit learn:
        http://scikit-learn.org/dev/index.html
        http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors

        Algorithm based on:
        Heyne, S., Costa, F., Rose, D., & Backofen, R. (2012).
        GraphClust: alignment-free structural clustering of local RNA secondary structures.
        Bioinformatics, 28(12), i224-i232.
        http://bioinformatics.oxfordjournals.org/content/28/12/i224.full.pdf+html"""
    def __init__(self, n_neighbors=5, radius=1.0, algorithm="approximate", number_of_hash_functions=400,
                 max_bin_size = 50, minimal_blocks_in_common = 1, block_size = 4, excess_factor = 5):
        self._inverseIndex = InverseIndex(number_of_hash_functions=number_of_hash_functions,
                                          max_bin_size =max_bin_size,
                                          number_of_nearest_neighbors=n_neighbors,
                                          minimal_blocks_in_common=minimal_blocks_in_common,
                                          block_size=block_size,
                                          excess_factor=excess_factor)
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self._X = None
        self._y = None
        self._sizeOfX = None
        self._shape_of_X = None

    def fit(self, X, y=None):
        """Fit the model using X as training data.

            Parameters
            ----------
            X : {array-like, sparse matrix}
                Training data. If array or matrix, shape = [n_samples, n_features]
            y : list, optional (default = None)
                List of classes for the given input of X. Size have to be n_samples."""
        X_csr = csr_matrix(X)
        self._X = X_csr
        self._sizeOfX = X_csr.shape[0]
        self._shape_of_X = X_csr.shape[1]
        self._y = y
        self._inverseIndex.fit(X_csr)

    def partial_fit(self, X, y=None):
        """Extend the model by X as additional training data.

            Parameters
            ----------
            X : {array-like, sparse matrix}
                Training data. Shape = [n_samples, n_features]
            y : list, optional (default = None)
                List of classes for the given input of X. Size have to be n_samples."""
        self._inverseIndex.partial_fit(X)
        # self._y
    def get_params(self,deep=None):
        """Get parameters for this estimator."""
        pass

    def kneighbors(self,X=None, n_neighbors=None, return_distance=True, algorithm=None):
        """Finds the n_neighbors of a point X or of all points of X.

            Parameters
            ----------
            X : {array-like, sparse matrix}, optional
                Data point(s) to be searched for n_neighbors. Shape = [n_samples, n_features]
                If not provided, neighbors of each indexed point are returned.
                In this case, the query point is not considered its own neighbor.
            n_neighbors : int, optional
                Number of neighbors to get (default is the value passed to the constructor).
            return_distance : boolean, optional. Defaults to True.
                If False, distances will not be returned
            algorithm : {'approximate', 'exact'}, optional (default = 'approximate')
            - 'approximate':    will only use an inverse index to compute a k_neighbor query.
            - 'exact':          an inverse index is used to preselect instances, and these are used to get
                                the original data from the data set to answer a k_neighbor query. The
                                original data is stored in the memory.
                                If not passed, default value is what was passed to the constructor.

            Returns
            -------
            dist : array
                Array representing the lengths to points, only present if
                return_distance=True
            ind : array
                Indices of the nearest points in the population matrix."""
        if algorithm is not None:
            if algorithm in ['approximate', 'exact']:
                self.algorithm = algorithm
            else:
                print "Algorithm not known. Choose between: 'approximate' or 'exact'."
                return
        if n_neighbors == None:
            n_neighbors = self.n_neighbors
        return self._neighborhood(X=X, neighborhood_measure=n_neighbors,
                                  return_distance=return_distance, computing_function="kneighbors")

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
            algorithm : {'approximate', 'exact'}, optional (default = 'approximate')
            - 'approximate':    will only use an inverse index to compute a k_neighbor query.
            - 'exact':          an inverse index is used to preselect instances, and these are used to get
                                the original data from the data set to answer a k_neighbor query. The
                                original data is stored in the memory.
                                If not passed, default value is what was passed to the constructor.
            Returns
            -------
            A : sparse matrix in CSR format, shape = [n_samples, n_samples_fit]
                n_samples_fit is the number of samples in the fitted data
                A[i, j] is assigned the weight of edge that connects i to j.
            """
        if algorithm is not None:
            if algorithm in ['approximate', 'exact']:
                self.algorithm = algorithm
            else:
                print "Algorithm not know. Choose between: 'approximate' or 'exact'."
                return
        if n_neighbors == None:
            n_neighbors = self.n_neighbors
            return_distance = True if mode == "connectivity" else False
        return self._neighborhood_graph(X=X, neighborhood_measure=n_neighbors, return_distance=return_distance,
                        computing_function="kneighbors")

    def radius_neighbors(self, X=None, radius=None, return_distance=None, algorithm=None):
        """Finds the neighbors within a given radius of a point or points.
        Return the indices and distances of each point from the dataset
        lying in a ball with size ``radius`` around the points of the query
        array. Points lying on the boundary are included in the results.
        The result points are *not* necessarily sorted by distance to their
        query point.
        Parameters
        ----------
        X : array-like, (n_samples, n_features), optional
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
        radius : float
            Limiting distance of neighbors to return.
            (default is the value passed to the constructor).
        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned
        algorithm : {'approximate', 'exact'}, optional (default = 'approximate')
            - 'approximate':    will only use an inverse index to compute a k_neighbor query.
            - 'exact':          an inverse index is used to preselect instances, and these are used to get
                                the original data from the data set to answer a k_neighbor query. The
                                original data is stored in the memory.
                                If not passed, default value is what was passed to the constructor.
        Returns
        -------
        dist : array, shape (n_samples,) of arrays
            Array representing the distances to each point, only present if
            return_distance=True. The distance values are computed according
            to the ``metric`` constructor parameter.
        ind : array, shape (n_samples,) of arrays
            An array of arrays of indices of the approximate nearest points
            from the population matrix that lie within a ball of size
            ``radius`` around the query points."""
        if algorithm is not None:
            self.algorithm = algorithm
        if radius == None:
            radius = self.radius
        return self._neighborhood(X=X, neighborhood_measure=radius,
                                  return_distance=return_distance, computing_function="radius_neighbors")

    def radius_neighbors_graph(self, X=None, radius=None, mode='connectivity', algorithm=None):
        """Computes the (weighted) graph of Neighbors for points in X
        Neighborhoods are restricted the points at a distance lower than
        radius.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features], optional
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
        radius : float
            Radius of neighborhoods.
            (default is the value passed to the constructor).
        mode : {'connectivity', 'distance'}, optional
            Type of returned matrix: 'connectivity' will return the
            connectivity matrix with ones and zeros, in 'distance' the
            edges are Euclidean distance between points.
        algorithm : {'approximate', 'exact'}, optional (default = 'approximate')
        - 'approximate':    will only use an inverse index to compute a k_neighbor query.
        - 'exact':          an inverse index is used to preselect instances, and these are used to get
                            the original data from the data set to answer a k_neighbor query. The
                            original data is stored in the memory.
                            If not passed, default value is what was passed to the constructor.
        Returns
        -------
        A : sparse matrix in CSR format, shape = [n_samples, n_samples]
        A[i, j] is assigned the weight of edge that connects i to j."""
        if algorithm is not None:
            self.algorithm = algorithm
        if radius == None:
            radius = self.radius
        return_distance = True if mode == "connectivity" else False
        return self._neighborhood_graph(X=X, neighborhood_measure=radius, return_distance=return_distance,
                            computing_function="radius_neighbors")

    def _neighborhood(self, X=None, neighborhood_measure=None, return_distance=None, computing_function=None):
        """Finds the K-neighbors of a point."""
        # define non local variables and functions as locals to increase performance
        inverseIndex = self._inverseIndex
        sizeOfX = self._sizeOfX
        build_reduced_neighborhood = self._build_reduced_neighborhood
        append = np.append
        asarray = np.asarray
        if X is None:
            X = self._X
            number_of_instances = self._X.shape[0]
            start_value = 1
        else:
            X = csr_matrix(X)
            start_value = 0
            number_of_instances = X.shape[0]
        distance_matrix = [[]] * number_of_instances
        kneighbors_matrix = [[]] * number_of_instances
        neighborhood_size = neighborhood_measure + start_value if computing_function == "kneighbors" else sizeOfX
        radius = neighborhood_measure
        # print "number_of_instances: ", number_of_instances
        # print "CPU count: ", mp.cpu_count()
        # intervalSize = int(ceil(number_of_instances / float(mp.cpu_count())))
        # j = 0
        # interval = []
        # while j < number_of_instances:
        #     interval.append(j)
        #     j += intervalSize
        # i = None
        # print "interval: ", interval
        # print "Type of slicing: ", type(X[0:5])
        # print "Size of interval: ", len(interval)
        # pool = mp.Pool(processes=len(interval))
        # results = [apply_async(pool, self._computeNeighborhood, args=(X[i:i+intervalSize], neighborhood_size, computing_function, neighborhood_measure, start_value)) for i in interval]
        # print "Result size: ", len(results)
        # print "Result: ", results
        # output = [p.get() for p in results]
        # print output

        # self, instance, neighborhood_size, computing_function, neighborhood_measure, index
        for i in xrange(number_of_instances):
            signature = inverseIndex.signature(X[i])
            if self.algorithm == "approximate":
                distances, neighbors = inverseIndex.neighbors(signature, neighborhood_size)
            else:
                distances, neighbors = build_reduced_neighborhood(X=X, computing_function=computing_function,
                                                             neighborhood_measure=neighborhood_measure,
                                                             signature=signature, return_distance = True,
                                                             index=i)
            if len(neighbors) == 0:
                    logger.warning("No matches in inverse index!")
            if computing_function == "radius_neighbors":
                for j in xrange(len(distances[0])):
                    if not distances[0][j] <= radius:
                        neighborhood_size = j
                        break
            else:
                appendSize = neighborhood_size - len(neighbors[0])
                if appendSize > 0:
                    valueNeighbors = [-1] * appendSize
                    valueDistances = [0] * appendSize
                    neighbors[0] = append(neighbors[0], valueNeighbors)
                    distances[0] = append(distances[0], valueDistances)

            kneighbors_matrix[i] = neighbors[0][start_value:neighborhood_size]
            distance_matrix[i] = distances[0][start_value:neighborhood_size]

        if return_distance:
            return asarray(distance_matrix), asarray(kneighbors_matrix)
        else:
            return asarray(kneighbors_matrix)

    def _build_reduced_neighborhood(self, X=None, computing_function=None,
                                    neighborhood_measure=None, signature = None,
                                    return_distance=None, index = None):
        # define non local variables and functions as locals to increase performance
        _X = self._X
        # if X is None, the element itself should not be returned. Otherwise it is considered with a distance = 0.
        start_value = 1 if X is None else 0
        # compute neighbors via inverse index
        distances, candidate_list = self._inverseIndex.neighbors(signature, neighborhood_measure)
        if len(candidate_list[0]) == 0:
            return distances, candidate_list
        # use sklearns implementation of NearestNeighbors to compute the neighbors
        # use only the elements of the candidate list and not the whole dataset
        nearest_neighbors = NearestNeighbors()
        nearest_neighbors.fit(_X[candidate_list[0]])

        if computing_function == "kneighbors":
            if len(candidate_list) < neighborhood_measure + start_value:
                neighborhood_measure = len(candidate_list[0]) - start_value
        if X is None:
            X = _X
            neighborhood_measure + start_value
        # compute kneighbors or the neighbors inside a given radius
        if computing_function == "kneighbors":
            distances, neighbors = nearest_neighbors.kneighbors(X[index], neighborhood_measure, return_distance)
        else:
            distances, neighbors = nearest_neighbors.radius_neighbors(X[index], neighborhood_measure, return_distance)
        # replace indices to the indicies of the orginal dataset
        for i in xrange(len(neighbors[0])):
            neighbors[0][i] = candidate_list[0][neighbors[0][i]]
        return distances.tolist(),  neighbors.tolist()

    def _neighborhood_graph(self, X=None, neighborhood_measure=None, return_distance=None,
                            computing_function=None):
        # choose if kneighbors or the neighbors in a given radius should be computed
        if computing_function == "kneighbors":
            distances, neighborhood = self.kneighbors(X, neighborhood_measure, True)
        elif computing_function == "radius_neighbors":
            distances, neighborhood = self.radius_neighbors(X, neighborhood_measure, True)
        else:
            logger.error("Computing function is not known!")
            return
        if len(neighborhood) == 0:
         pass
        # build the neighborhood graph
        row = []
        col = []
        data = []
        start_value = 0 if X is None else 1
        for i in xrange(len(neighborhood)):
            root = i if X is None else neighborhood[i][0]
            for node in neighborhood[i][start_value:]:
                row.append(root)
                col.append(node)
            data.extend(distances[i][start_value:])
        # if the distances do not matter overrite them with 1's.
        if return_distance:
            data = [1] * len(row)
        return csr_matrix((data, (row, col)))

    def _computeNeighborhood(self, X, neighborhood_size, computing_function, neighborhood_measure, start_value):

        # define non-local variables as local ones to increase performance
        inverseIndex = self._inverseIndex
        sizeOfX = self._sizeOfX
        build_reduced_neighborhood = self._build_reduced_neighborhood
        append = np.append
        asarray = np.asarray
        number_of_instances = X.shape[0]
        print "Length of data: ", number_of_instances
        distance_matrix = [[]] * number_of_instances
        kneighbors_matrix = [[]] * number_of_instances
        for i in xrange(number_of_instances):
            # compute the signature and get the nearest neighbors from the inverse index.
            signature = inverseIndex.signature(X[i])
            if self.algorithm == "approximate":
                distances, neighbors = inverseIndex.neighbors(signature, neighborhood_size)
            else:
                distances, neighbors = build_reduced_neighborhood(X=X[i], computing_function=computing_function,
                                                             neighborhood_measure=neighborhood_measure,
                                                             signature=signature, return_distance = True,
                                                             index=i)
            if len(neighbors) == 0:
                    logger.warning("No matches in inverse index!")
            if computing_function == "radius_neighbors":
                for j in xrange(len(distances[0])):
                    if not distances[0][j] <= radius:
                        neighborhood_size = j
                        break
            else:
                # if the number of neighbors avaible is less then the defined number of neighbors
                # that should be returned, add for the indices "-1"'s and for the distance 0's
                appendSize = neighborhood_size - len(neighbors[0])
                if appendSize > 0:
                    valueNeighbors = [-1] * appendSize
                    valueDistances = [0] * appendSize
                    neighbors[0] = append(neighbors[0], valueNeighbors)
                    distances[0] = append(distances[0], valueDistances)

        # neighbors[0][start_value:neighborhood_size], distances[0][start_value:neighborhood_size]
        # cut the neighborhood [1:neighbors+1]  or [0:n_neigbors] 
        # depends on if X == None or not
        kneighbors_matrix[i] = neighbors[0][start_value:neighborhood_size]
        distance_matrix[i] = distances[0][start_value:neighborhood_size]
        return distance_matrix, kneighbors_matrix
        # return distance_matrix
    def set_params(**params):
        pass