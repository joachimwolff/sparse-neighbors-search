# Copyright 2015 Joachim Wolff
# Master Thesis
# Tutors: Milad Miladi, Fabrizio Costa
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
    def fit(self, X, y=None, saveMemory=None, pThreads=None):

        if saveMemory:
            print('partial fitting')

            if pThreads and pThreads > 1:
                pass
            else:
                self._minHashObject.fit(X.getrow(0))
                for i in range(1, X.shape[0]):
                    self._minHashObject.partial_fit(X.getrow(i))
            print('partial fitting ...DONE')
            
        else:
            print('full fit')
            self._minHashObject.fit(X)
        self._precomputed_graph = self._minHashObject.kneighbors_graph(mode='distance')
        self._clusteringObject.fit(self._precomputed_graph)
	
    def fit_predict(self, X, y=None, saveMemory=None):

        self.fit(X, y, saveMemory=saveMemory)
        # self.fit(X, y, saveMemory=False)

        return self.predict(self._precomputed_graph, y)
		
    def predict(self, X, y=None):
        if hasattr(self._clusteringObject, 'labels_'):
            return self._clusteringObject.labels_.astype(np.int)
        else:
            return self._clusteringObject.predict(X)