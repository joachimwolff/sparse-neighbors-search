# Copyright 2016, 2017, 2018, 2019, 2020 Joachim Wolff
# PhD Thesis
#
# Copyright 2015, 2016 Joachim Wolff
# Master Thesis
# Tutor: Fabrizio Costa
# Winter semester 2015/2016
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwigs-University Freiburg im Breisgau

from ..neighbors import MinHash
import numpy as np
from scipy.sparse import vstack

class MinHashClustering():
    def __init__(self, minHashObject, clusteringObject):
        self._minHashObject = minHashObject
        self._clusteringObject = clusteringObject
        self._precomputed_graph = None
    def fit(self, X, y=None, pSaveMemory=None):
        if pSaveMemory is not None and pSaveMemory > 0:
            if pSaveMemory > 1:
                pSaveMemory = 1
            number_of_elements = X.shape[0]
            batch_size = int(np.floor(number_of_elements * pSaveMemory))
            if batch_size < 1:
                batch_size = 1
            self._minHashObject.fit(X[0:batch_size, :])
            if batch_size < number_of_elements:
                for i in range(batch_size, X.shape[0], batch_size):
                    self._minHashObject.partial_fit(X[i:i+batch_size, :])
        else:
            self._minHashObject.fit(X)
        self._precomputed_graph = self._minHashObject.kneighbors_graph(mode='distance')
        self._clusteringObject.fit(self._precomputed_graph)
	
    def fit_predict(self, X, y=None, pSaveMemory=None):

        self.fit(X, y, pSaveMemory=pSaveMemory)

        return self.predict(self._precomputed_graph, y)
		
    def predict(self, X, y=None):
        if hasattr(self._clusteringObject, 'labels_'):
            return self._clusteringObject.labels_.astype(np.int)
        else:
            return self._clusteringObject.predict(X)