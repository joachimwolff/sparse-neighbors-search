# Copyright 2015 Joachim Wolff
# Master Thesis
# Tutors: Milad Miladi, Fabrizio Costa
# Winter semester 2015/2016
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwig-University Freiburg im Breisgau

__author__ = 'joachimwolff'
import logging
import multiprocessing as mp
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from sklearn.utils import check_X_y

from numpy import array

import math
import _minHash

logger = logging.getLogger(__name__)

class MinHash():
    """Approximate unsupervised learner for implementing neighbor searches on sparse data sets. Based on a
        dimension reduction with minimum hash functions.

        Parameters
        ----------
        n_neighbors : int, optional (default = 5)
            Number of neighbors to use by default for :meth:`k_neighbors` queries.
        radius : float, optional (default = 1.0)
            Range of parameter space to use by default for :meth`radius_neighbors`
            queries.

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
            all cores are getting the same amount of data at once; e.g. 8-core cpu and 128 elements to process, every core will
            get 16 elements at once.
        
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
    def __init__(self, n_neighbors=5, radius=1.0, fast=False, number_of_hash_functions=400,
                 max_bin_size = 50, minimal_blocks_in_common = 1, block_size = 4, excess_factor = 5,
                 number_of_cores=None, chunk_size=None):
        if number_of_cores is None:
            number_of_cores = mp.cpu_count()
        if chunk_size is None:
            chunk_size = 0
        maximal_number_of_hash_collisions = int(math.ceil(number_of_hash_functions / float(block_size)))
        _index_elements_count = 0
        self._pointer_address_of_minHash_object = _minHash.createObject(number_of_hash_functions, 
                                                    block_size, number_of_cores, chunk_size, n_neighbors,
                                                    minimal_blocks_in_common, max_bin_size, 
                                                    maximal_number_of_hash_collisions, excess_factor,
                                                    1 if fast else 0)

    def fit(self, X, y=None):
        """Fit the model using X as training data.

            Parameters
            ----------
            X : {array-like, sparse matrix}, optional
                Training data. If array or matrix, shape = [n_samples, n_features]
                If X is None, a "lazy fitting" is performed. If kneighbors is called, the fitting
                with with the data there is done. Also the caching of computed hash values is deactivated in
                this case.
            y : list, optional (default = None)
                List of classes for the given input of X. Size have to be n_samples."""
        # if y is not None:
        #     self._y_is_csr = True
        #     self._X, self._y = check_X_y(X, y, "csr", multi_output=True)
        #     if self._y.ndim == 1 or self._y.shape[1] == 1:
        #         self._y_is_csr = False
        # else:
        #     self._X = csr_matrix(X)
        #     self._y_is_csr = False
        
        # self._sizeOfX = self._X.shape[0]
        # self._shape_of_X = self._X.shape[1]
        # self._inverseIndex.fit(self._X)

        self._index_elements_count = X.shape[0]
        instances, features = X.nonzero()
        data = X.data
        # returns a pointer to the inverse index stored in c++
        self._pointer_address_of_minHash_object = _minHash.fit(instances.tolist(), features.tolist(), data.tolist(), 
                                                    X.shape[0], X.shape[1],
                                                    self._pointer_address_of_minHash_object)
        

    def partial_fit(self, X, y=None):
        """Extend the model by X as additional training data.

            Parameters
            ----------
            X : {array-like, sparse matrix}
                Training data. Shape = [n_samples, n_features]
            y : list, optional (default = None)
                List of classes for the given input of X. Size have to be n_samples."""
        instances, features = X.nonzero()
        data = X.data
        for i in xrange(len(instances)):
            instances[i] += self._index_elements_count 
        self._index_elements_count  += X.shape[0]
        self._pointer_address_of_minHash_object = _minHash.fit(instances.tolist(), features.tolist(), data.tolist(),
                                            self._pointer_address_of_minHash_object)
       
        
    def kneighbors(self,X=None, n_neighbors=None, return_distance=True, fast=None):
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
            dist : array, shape = [n_samples, distances]
                Array representing the lengths to points, only present if
                return_distance=True
            ind : array, shape = [n_samples, neighbors]
                Indices of the nearest points in the population matrix."""
        max_number_of_instances = 0
        max_number_of_features = 0
        if X is None:
            print "Starting PYTHON"

            bla= _minHash.kneighbors([], [], [], 
                                    0, 0,
                                    n_neighbors if n_neighbors else 0,
                                    1 if return_distance else 0,
                                    1 if fast else 0, self._pointer_address_of_minHash_object)
            print "END PYTHON"

        else:
            instances, features = X.nonzero()
            data = X.data()
            max_number_of_instances = X.shape[0]
            max_number_of_features = X.shape[1]
            print "Starting PYTHON"
            bla =  _minHash.kneighbors(instances.tolist(), features.tolist(), data.tolist(), 
                                    max_number_of_instances, max_number_of_features,
                                    n_neighbors if n_neighbors else 0,
                                    1 if return_distance else 0,
                                    1 if fast else 0, self._pointer_address_of_minHash_object)
            print "END PYTHON"
        print bla


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

        if X is None:
            instances = array(-1)
            features = array(-1)
            data = array(-1)
        else:
            instances, features = X.nonzero()
            data = X.data()

        mode_cpp = -1
        if mode == "connectivity":
            mode_cpp = 0
        elif mode == 'distance':
            mode_cpp = 1
        if mode_cpp == -1:
            return
        return _minHash.kneighbors(instances.tolist(), features.tolist(), data.tolist(), 
                                    n_neighbors if n_neighbors else -1,
                                    mode_cpp,
                                    1 if fast else 0)


    def radius_neighbors(self, X=None, radius=None, return_distance=None, fast=None):
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
        if X is None:
            instances = array(-1)
            features = array(-1)
            data = array(-1)
        else:
            instances, features = X.nonzero()
            data = X.data()

        mode_cpp = -1
        if mode == "connectivity":
            mode_cpp = 0
        elif mode == 'distance':
            mode_cpp = 1
        if mode_cpp == -1:
            return
        return _minHash.radius_neighbors(instances.tolist(), features.tolist(), data.tolist(), 
                                    radius if radius else -1,
                                    1 if return_distance else 0,
                                    1 if fast else 0)

    def radius_neighbors_graph(self, X=None, radius=None, mode='connectivity', fast=None):
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
        if fast is not None:
            self.fast = fast
        else:
            self.fastComputation = self._fast
        if radius == None:
            radius = self.radius
        return_distance = True if mode == "connectivity" else False
        return self._neighborhood_graph(X=X, neighborhood_measure=radius, return_distance=return_distance,
                            computing_function="radius_neighbors")


    def fit_kneighbors(self, X, n_neighbors=None, return_distance=True, fast=None):
        """"Fits and returns the n_neighbors of X.

        Parameters
            ----------
            X : {array-like, sparse matrix}
                Data point(s) to be fitted and searched for n_neighbors. Shape = [n_samples, n_features]
            y : list, optional (default = None)
                List of classes for the given input of X. Size have to be n_samples.
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
            dist : array, shape = [n_samples, distances]
                Array representing the lengths to points, only present if
                return_distance=True
            ind : array, shape = [n_samples, neighbors]
                Indices of the nearest points in the population matrix."""
        self.fit(X)
        if fast is not None:
            self.fastComputation = fast
        else:
            self.fastComputation = self._fast
        if n_neighbors == None:
            n_neighbors = self.n_neighbors
        return self._neighborhood(X=X, neighborhood_measure=n_neighbors,
                                  return_distance=return_distance, computing_function="kneighbors",  lazy_fitting=True)
        

    def fit_kneighbor_graph(self, X, n_neighbors=None, mode='connectivity', fast=None):
        """Fits and computes the (weighted) graph of k-Neighbors for points in X
            Parameters
            ----------
            X : array-like, last dimension same as that of fit data
                The query point or points.
            y : list, optional (default = None)
                List of classes for the given input of X. Size have to be n_samples.   
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
        self.fit(X)
        if fast is not None:
            self.fast = fast
        else:
            self.fastComputation = self._fast
        if n_neighbors == None:
            n_neighbors = self.n_neighbors
        return_distance = True if mode == "connectivity" else False
        return self._neighborhood_graph(X=X, neighborhood_measure=n_neighbors, return_distance=return_distance,
                        computing_function="kneighbors")
        return self.kneighbors_graph(X, n_neighbors=n_neighbors, mode=mode, fast=fast)

    def fit_radius_neighbors(self, X, y=None, radius=None, return_distance=None, fast=None):
        """Finds the neighbors within a given radius of a point or points.
        Return the indices and distances of each point from the dataset
        lying in a ball with size ``radius`` around the points of the query
        array. Points lying on the boundary are included in the results.
        The result points are *not* necessarily sorted by distance to their
        query point.
        Parameters
        ----------
        X : array-like, (n_samples, n_features)
            The to be fitted data and query point or points.
        y : list, optional (default = None)
                List of classes for the given input of X. Size have to be n_samples.   
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
        self.fit(X, y)
        return self.radius_neighbors(X, radius=radius, return_distance=return_distance, fast=fast)
        
    def fit_radius_neighbors_graph(self, X, y=None, radius=None, mode='connectivity', fast=None):
        """Fits and computes the (weighted) graph of Neighbors for points in X
        Neighborhoods are restricted the points at a distance lower than
        radius.

        Parameters
        ----------
        X : array-like, (n_samples, n_features)
            The to be fitted data and query point or points.
        y : list, optional (default = None)
            List of classes for the given input of X. Size have to be n_samples.      
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
        self.fit(X, y)
        return self.radius_neighbors_graph(X, radius=radius, mode=mode, fast=fast)

    def _neighborhood(self, X=None, neighborhood_measure=None, return_distance=None, computing_function=None, lazy_fitting=False):
        """Controlls the search of the nearest neighbors. It depends on the the choosen algorithm if the
            approximate or exact version is running and on the parameter \"computing_function\" if K-nearest_neighbor_algorithm
            or the radius_neighbors_algorithm is executed.
            """
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
        if lazy_fitting:
            start_value = 1
        distance_matrix = [[]] * number_of_instances
        kneighbors_matrix = [[]] * number_of_instances
        neighborhood_size = neighborhood_measure + start_value if computing_function == "kneighbors" else sizeOfX
        radius = neighborhood_measure

        if self.fastComputation:
            distances, neighbors = inverseIndex.neighbors(X, neighborhood_size, lazy_fitting)
        else:
            distances, neighbors = build_reduced_neighborhood(X=X, computing_function=computing_function,
                                                             neighborhood_measure=neighborhood_size,
                                                             return_distance = True, lazy_fitting=lazy_fitting)
        for i in xrange(number_of_instances):
            if len(neighbors) == 0:
                    logger.warning("No matches in inverse index!")
            if computing_function == "radius_neighbors":
                for j in xrange(len(distances[i][0])):
                    if not distances[i][0][j] <= radius:
                        neighborhood_size = j
                        break
            else:
                #print "Index if list: ", i
                if len(neighbors) == 0:
                    logger.warning("No matches in inverse index!")
                appendSize = neighborhood_size - len(neighbors[i])
                if appendSize > 0:
                    valueNeighbors = [-1] * appendSize
                    valueDistances = [0] * appendSize
                    neighbors[i] = append(neighbors[i], valueNeighbors)
                    distances[i] = append(distances[i], valueDistances)

            kneighbors_matrix[i] = neighbors[i][start_value:neighborhood_size]
            distance_matrix[i] = distances[i][start_value:neighborhood_size]

        if return_distance:
            return asarray(distance_matrix), asarray(kneighbors_matrix)
        else:
            return asarray(kneighbors_matrix)

    def _build_reduced_neighborhood(self, X=None, computing_function=None,
                                    neighborhood_measure=None,
                                    return_distance=None, lazy_fitting=None):
        """This function is computing the \"exact\"-version of the algorithm.
            It takes all possible candidates for the neighbor search out of the
            inverse index, takes the orginal data for these instances and is putting
            this reduced dataset into the the sklearn implementation."""
        # define non local variables and functions as locals to increase performance
        _X = self._X
        inverseIndex = self._inverseIndex

        distances, neighbors = inverseIndex.neighbors(X, neighborhood_measure, lazy_fitting)

        real_index__candidate_list_index = [{}] * len(neighbors)
        for i in xrange(len(neighbors)):
            if len(neighbors[i]) > 1:
                dict_ = {}
                for j in xrange(len(neighbors[i])):
                    dict_[j] = neighbors[i][j]
                real_index__candidate_list_index[i] = dict_
       
        for i in xrange(X.shape[0]):
            if len(neighbors[i]) > 1:
                innerproduct = X[[i]] * _X[neighbors[i]].transpose()
                neighbor_list = np.argsort(innerproduct.toarray())[0][::-1] 
                for j in xrange(len(neighbor_list)):
                    distances[i][j] = innerproduct[0,neighbor_list[j]]
                    neighbors[i][j] = real_index__candidate_list_index[i][neighbor_list[j]]
        return distances, neighbors

    def _neighborhood_graph(self, X=None, neighborhood_measure=None, return_distance=None,
                            computing_function=None):
        """This function calls kneighbors respectively radius_neighbors and creates 
            out of the returned data a graph."""
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
            if distances[i][0] == -1:
               continue
            root = i if X is None else neighborhood[i][0]
            j = 0
            for node in xrange(len(neighborhood[i][start_value:])):
                if distances[i][node] == -1:
                    end_value = j
                    break
                if root < 0 or neighborhood[i][node] < 0:
                    end_value = j
                    break;
                row.append(root)
                col.append(neighborhood[i][node])
                j += 1
            if j > 0 and  j < len(neighborhood[i][start_value:]):
                data.extend(distances[i][start_value:end_value])
            elif len(neighborhood[i][start_value:]) > j:
                data.extend(distances[i][start_value:])
        # if the distances do not matter overrite them with 1's.
        if return_distance:
            data = [1] * len(row)

        return csr_matrix((data, (row, col)))