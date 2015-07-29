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

import math
import logging
from _hashUtility import computeInverseIndex
from _hashUtility import computeSignature

logger = logging.getLogger(__name__)

class InverseIndex():
    """The inverse index computes and stores all hash collisions.
    Parameters
    ----------
    number_of_hash_functions : int, optional (default = 400)
        Number of hash functions to compute the minHash value. With more hash function the error rate decreases
        and the probalbility of hash collisions increases.
    max_bin_size : int, optional (default = 50)
        Number of maximal accepted hash collisions.
    number_of_nearest_neighbors : int, optional (default = 5)
        Number of nearest neighbors that should be returned.
    minimal_blocks_in_common : int, optional (default = 1)
        Number of minimal collisions two instances should have to be recognised.
    block_size : int, optional (default = 4)
        Factor to reduces the size of one signature. The size of a signature is given by:
        number_of_hash_functions / block_size. block_size have to be smaller than number_of_hash_functions
    excess_factor: int, optional (default = 5)
        Factor to increase the number of returned neighbors. Useful for the 'exact' algorithm; has no influence
        for the approximate solution.
    """
    def __init__(self, number_of_hash_functions=400,
                 max_bin_size = 50,
                 number_of_nearest_neighbors = 5,
                 minimal_blocks_in_common = 1,
                 block_size = 4,
                 excess_factor = 5, 
                 number_of_cores = -1,
                 chunk_size = 0):
        self._number_of_hash_functions = number_of_hash_functions
        self._max_bin_size = max_bin_size
        self._number_of_nearest_neighbors = number_of_nearest_neighbors
        self._minimal_blocks_in_common = minimal_blocks_in_common
        self._block_size = block_size
        if self._number_of_hash_functions < self._block_size:
            logger.error("Number of hash functions is smaller than the block size!")
        if int(math.ceil(self._number_of_hash_functions / float(self._block_size))) < 1:
            logger.error("The allowed number of hash collisions is smaller than 1. Please increase number_of_hash_functions"
                         "or decrease the block_size.")
        self._maximal_number_of_hash_collisions = int(math.ceil(self._number_of_hash_functions / float(self._block_size)))
        self._inverse_index = None
        self._excess_factor = excess_factor
        self._signature_storage = {}
        self.MAX_VALUE = 2147483647
        self._number_of_cores = number_of_cores
        self._chunk_size = chunk_size

    def fit(self, X):
        """Fits the given dataset to the minHash model.

        Parameters
        ----------
        X : csr_matrix
            The input matrix to compute the inverse index. If empty, no index will be created.
        """
        # get all ids of non-null features per instance and compute the inverse index in c++.
        instances, features = X.nonzero()
        self._inverse_index = computeInverseIndex(self._number_of_hash_functions,
                                instances.tolist(), features.tolist(), self._block_size, self._max_bin_size,
                                                  self._number_of_cores, self._chunk_size)

    def partial_fit(self, X):
        """Extends the inverse index build in 'fit'.

        Parameters
        ----------
        X : csr_matrix
            The input matrix to compute the inverse index. If empty, no index will not be extended.
        """
        # extend the inverse index with the given data
        index = len(self._signature_storage)
        for i in xrange(0, X.get_shape()[0]):
            signature = self.signature(X[[i]])
            self._update_inverse_index(signature, index)
            index += 1

    def signature(self, instance_feature_list):
        """Computes the signature based on the minHash model for one instaces.

        Parameters
        ----------
        instance : csr_matrix
            The input row to compute the signature.
        """


        instances, features = instance_feature_list.nonzero()
        # compute the siganture in c++
        a = computeSignature(self._number_of_hash_functions,instances.tolist(), features.tolist() ,
                                                self._block_size, self._number_of_cores, self._chunk_size)
        return a


    def _update_inverse_index(self, signature, index):
        """This function extends the inverse index given a specific signature and the id.
            If an instance is already available, the inverse index is extended. The old values of
            this instance are not deleted.

        Parameters
        ----------
        signature : list
            List of hash values. Index of a value corresponds to a hash function.
        index : int
            ID of the instance for the given signature.
        """
        inverse_index = self._inverse_index
        max_bin_size = self._max_bin_size
        for i in xrange(0, len(signature)):
            try:
                # add the instance for the given hash value of a hash function
                # only if for this hash value not more than max_bin_size-instances
                # are stored.
                if signature[i] in inverse_index[i]:
                    if len(inverse_index[i][signature[i]]) < max_bin_size:
                        if inverse_index[i][signature[i]] != [-1]:
                            inverse_index[i][signature[i]].append(index)
                    else:
                        inverse_index[i][signature[i]] = [-1]
                        logging_string = "Too much hash collisions. Hashfunction: ",i," value: ", signature[i], " is not considered."
                        logger.info(logging_string)
                else:
                    # for the given hash value of the given hash function no 
                    # instance is stored
                    inverse_index[i][signature[i]] = [index]
            except:
                logger.error("Inverse index can not be updated!")


    def neighbors(self, signature, size_of_neighborhood):
        """This function computes the neighborhood for a given instance.

        Parameters
        ----------
        signature : list
            List of hash values. Index of a value corresponds to a hash function.
        size_of_neighborhood : int
            Size of neighborhood. If less neighbors are av
        """

        # define non local variables and functions as locals to increase performance
        inverse_index = self._inverse_index
        max_bin_size = self._max_bin_size
        minimal_blocks_in_common = self._minimal_blocks_in_common
        MAX_VALUE = self.MAX_VALUE
        neighborhood = {}
        for i in xrange (0, len(signature)):
            hash_id = signature[i]
            if hash_id != 0 and hash_id != MAX_VALUE:
                # check the inverse index if for the given hash function 
                # the specific hash value is avaible.
                if hash_id in inverse_index[i]:
                    collision_size = len(inverse_index[i][hash_id])
                else:
                    continue
                if collision_size < max_bin_size:
                    for instance_id in inverse_index[i][hash_id]:
                        if instance_id in neighborhood:
                            neighborhood[instance_id] += 1
                        else:
                            neighborhood[instance_id] = 1
        # sort the instances by hits
        neighborhood_inverse = {}
        for i in neighborhood:
            if neighborhood[i] >= minimal_blocks_in_common:
                neighborhood_inverse[-neighborhood[i]] = i
        neighborhood_sorted = sorted(neighborhood_inverse.iteritems(), key=lambda key_value: key_value[0])
        neighborhood_list = []
        distance_list = []
        # compute distance with collisions: 1 - (collisions/max_number_of_collsions)
        for i in xrange(len(neighborhood_sorted)):
            neighborhood_list.append(neighborhood_sorted[i][1])
            distance = 1 - (abs(neighborhood_sorted[i][0]) / float(self._maximal_number_of_hash_collisions))
            distance_list.append(distance)
        return self._trim_neighborhood(neighborhood=neighborhood_list, distances=distance_list,
                                    size_of_neighborhood=size_of_neighborhood)

    def _trim_neighborhood(self, neighborhood=None, size_of_neighborhood=None, distances=None):
        """
        Trims the neighborhood to which is smaller: The number of found 
        neighbors or the desirable size of the neighborhood * the excess factor.
    
        Parameters
        ----------
        neighborhood : list
            The list of found neighbors.
        size_of_neighborhood : int
            Desirable size of the neighborhood.
        distances : list
            The distances from the target to the found neighbors.

        """
        size_of_neighborhood_adjusted = min((size_of_neighborhood*self._excess_factor) , len(neighborhood))
        return [distances[0:size_of_neighborhood_adjusted]], [neighborhood[0:size_of_neighborhood_adjusted]]

    def get_signature_list(self, instances):
        """Returns all signatures for the given sequences."""
        siganture_list = [[]] * instances.shape[0]
        for i in xrange(instances.shape[0]):
            siganture_list[i] = self.signature(instances[i])
        return siganture_list