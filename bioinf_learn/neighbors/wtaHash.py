# Copyright 2016 Joachim Wolff
# Master Thesis
# Tutor: Fabrizio Costa, Milad Miladi
# Winter semester 2015/2016
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwigs-University Freiburg im Breisgau

__author__ = 'joachimwolff'
from scipy.sparse import csr_matrix


from nearestNeighborsCppInterface import NearestNeighborsCppInterface

class WtaHash():
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
        shingle_size : int, optional (default = 4)
            Reduction factor for the signature size.
            E.g. number_of_hash_functions=400 and shingle_size=4 --> Size of the signature will be 100
        excess_factor : int, optional (default = 5)
            Factor to return more neighbors internally as defined with n_neighbors. Factor is useful to increase the
            precision of the :meth:`algorithm=exact` version of the implementation.
            E.g.: n_neighbors = 5, excess_factor = 5. Internally n_neighbors*excess_factor = 25 neighbors will be returned.
            Now the reduced data set for sklearn.NearestNeighbors is of size 25 and not 5.
        similarity : bool, optional
            If true: cosine similarity is used
            If false: Euclidean distance is used
        number_of_cores : int, optional
            Number of cores that should be used for openmp. If your system doesn't support openmp, this value
            will have no effect. If it supports openmp and it is not defined, the maximum number of cores is used.
        chunk_size : int, optional
            Number of elements one cpu core should work on. If it is set to "0" the default behaviour of openmp is used;
            all cores are getting the same amount of data at once; e.g. 8-core cpu and 128 elements to process, every core will
            get 16 elements at once.
        prune_inverse_index=-1, Remove every hash value with less occurence than prune_inverse_index. if -1 it is deactivated
        prune_inverse_index_after_instance=-1.0, Start all the pruning routines after x% of the fitting process
        remove_hash_function_with_less_entries_as=-1, Remove every hash function with less hash values as n
        block_size = 5, 
        shingle=0,
        store_value_with_least_sigificant_bit=0
        cpu_gpu_load_balancing 0 if 100% cpu, 1 if 100% gpu. if e.g. 0.7 it means 70% gpu, 30% cpu
        
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
    def __init__(self, n_neighbors = 5, radius = 1.0, fast=False, number_of_hash_functions=400, rangeK_wta=20,
                 max_bin_size = 50, minimal_blocks_in_common = 1, shingle_size = 4, excess_factor = 5,
                 similarity=False, number_of_cores=None, chunk_size=None, prune_inverse_index=-1,
                 prune_inverse_index_after_instance=-1.0, remove_hash_function_with_less_entries_as=-1, 
                 block_size = 5, shingle=0, store_value_with_least_sigificant_bit=0, 
                 cpu_gpu_load_balancing=0, speed_optimized=None, accuracy_optimized=None):
                  
        if speed_optimized is not None and accuracy_optimized is not None:
            print "Speed optimization and accuracy optimization at the same time is not possible."
            return
        if speed_optimized:
            number_of_hash_functions = 400
            max_bin_size = 12
            minimal_blocks_in_common = 1
            shingle_size = 1
            excess_factor = 12
            prune_inverse_index = 2
            prune_inverse_index_after_instance = 0.5
            remove_hash_function_with_less_entries_as = 0
            block_size = 1
            shingle = 1
            store_value_with_least_sigificant_bit = 4
        elif accuracy_optimized:
            number_of_hash_functions = 400
            max_bin_size = 12
            minimal_blocks_in_common = 1
            shingle_size = 1
            excess_factor = 12
            prune_inverse_index = 2
            prune_inverse_index_after_instance = 0.5
            remove_hash_function_with_less_entries_as = 0
            block_size = 1
            shingle = 1
            store_value_with_least_sigificant_bit = 4
            
        self._nearestNeighborsCppInterface = NearestNeighborsCppInterface(n_neighbors=n_neighbors, radius=radius,
                fast=fast, number_of_hash_functions=number_of_hash_functions,
                max_bin_size=max_bin_size, minimal_blocks_in_common=minimal_blocks_in_common,
                shingle_size=shingle_size, excess_factor=excess_factor,
                similarity=similarity, number_of_cores=number_of_cores, chunk_size=chunk_size, prune_inverse_index=prune_inverse_index,
                prune_inverse_index_after_instance=prune_inverse_index_after_instance,
                remove_hash_function_with_less_entries_as=remove_hash_function_with_less_entries_as, 
                hash_algorithm=1, block_size=block_size, shingle=shingle,
                store_value_with_least_sigificant_bit=store_value_with_least_sigificant_bit, 
                cpu_gpu_load_balancing=cpu_gpu_load_balancing, gpu_hashing=0, rangeK_wta=rangeK_wta)

    def __del__(self):
       del self._nearestNeighborsCppInterface
           

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
        self._nearestNeighborsCppInterface.fit(X=X, y=y)
        

    def partial_fit(self, X, y=None):
        """Extend the model by X as additional training data.

            Parameters
            ----------
            X : {array-like, sparse matrix}
                Training data. Shape = [n_samples, n_features]
            y : list, optional (default = None)
                List of classes for the given input of X. Size have to be n_samples."""
        self._nearestNeighborsCppInterface.partial_fit(X=X, y=y)
       
        
    def kneighbors(self,X=None, n_neighbors=None, return_distance=True, fast=None, similarity=None):
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
        return self._nearestNeighborsCppInterface.kneighbors(X=X, n_neighbors=n_neighbors, 
                                                                return_distance=return_distance,
                                                                fast=fast, similarity=similarity)

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity', fast=None, symmetric=True, similarity=None):
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
        return self._nearestNeighborsCppInterface.kneighbors_graph(X=X, n_neighbors=n_neighbors, mode=mode, 
                                                            fast=fast, symmetric=symmetric, similarity=similarity)

    def radius_neighbors(self, X=None, radius=None, return_distance=None, fast=None, similarity=None):
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
        fast : bool, optional (default = 'False')
            - 'True':    will only use an inverse index to compute a k_neighbor query.
            - 'False':          an inverse index is used to preselect instances, and these are used to get
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
        return self._nearestNeighborsCppInterface.radius_neighbors(X=X, radius=radius, 
                                                                    return_distance=return_distance, 
                                                                    fast=fast, similarity=similarity)
        

    def radius_neighbors_graph(self, X=None, radius=None, mode='connectivity', fast=None, symmetric=True, similarity=None):
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
        return self._nearestNeighborsCppInterface.radius_neighbors_graph(X=X, radius=radius, mode=mode,
                                                                            fast=fast, symmetric=symmetric, similarity=similarity)


    def fit_kneighbors(self, X, n_neighbors=None, return_distance=True, fast=None, similarity=None):
        """"Fits and returns the n_neighbors of X.

        Parameters
            ----------
            X : {array-like, sparse matrix}
                Data point(s) to be fitted and searched for n_neighbors. Shape = [n_samples, n_features]
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
        return self._nearestNeighborsCppInterface.fit_kneighbors(X=X, n_neighbors=n_neighbors,
                                                                    return_distance=return_distance,
                                                                    fast=fast, similarity=similarity)

    def fit_kneighbor_graph(self, X, n_neighbors=None, mode='connectivity', fast=None, symmetric=True, similarity=None):
        """Fits and computes the (weighted) graph of k-Neighbors for points in X
            Parameters
            ----------
            X : array-like, last dimension same as that of fit data
                The query point or points.
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
        return self._nearestNeighborsCppInterface.fit_kneighbor_graph(X=X, n_neighbors=n_neighbors,
                                                                        mode=mode, fast=fast, 
                                                                        symmetric=symmetric, 
                                                                        similarity=similarity)

    def fit_radius_neighbors(self, X, radius=None, return_distance=None, fast=None, similarity=None):
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
        return self._nearestNeighborsCppInterface.fit_radius_neighbors(X=X, radius=radius, 
                                                                        return_distance=return_distance,
                                                                        fast=fast, similarity=similarity)  
        
    def fit_radius_neighbors_graph(self, X, radius=None, mode='connectivity', fast=None, symmetric=True, similarity=None):
        """Fits and computes the (weighted) graph of Neighbors for points in X
        Neighborhoods are restricted the points at a distance lower than
        radius.

        Parameters
        ----------
        X : array-like, (n_samples, n_features)
            The to be fitted data and query point or points.   
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
        return self._nearestNeighborsCppInterface.fit_radius_neighbors_graph(X=X, radius=radius,
                                                                                mode=mode, fast=fast,
                                                                                symmetric=symmetric,
                                                                                similarity=similarity)
        
        
    def get_distribution_of_inverse_index(self):
        """Returns the number of created hash values per hash function, 
            the average size of elements per hash value per hash function,
            the mean and the standard deviation."""
        return self._nearestNeighborsCppInterface.get_distribution_of_inverse_index()
        
    def _getY(self):
        return self._nearestNeighborsCppInterface._getY()
    def _getY_is_csr(self):
        return self._nearestNeighborsCppInterface._getY_is_csr()