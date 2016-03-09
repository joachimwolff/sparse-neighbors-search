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

class MinHashClustering():
    def __init__(self, minHashObject, clusteringObject):
        self._minHashObject = minHashObject
        self._clusteringObject = clusteringObject
        
    def fit(self, X, y=None):
        self._minHashObject.fit(X)
        precomputed_graph = self._minHashObject.kneighbors_graph(mode='distance')
        self._clusteringObject.fit(precomputed_graph)
	
    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X, y)
		
    def predict(self, X, y=None):
        if hasattr(self._clusteringObject, 'labels_'):
            return self._clusteringObject.labels_.astype(np.int)
        else:
            return self._clusteringObject.predict(X)