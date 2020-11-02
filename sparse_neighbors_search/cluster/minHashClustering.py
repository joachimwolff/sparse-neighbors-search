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
from sklearn.decomposition import PCA
# from scanpy import tl, pp
# from anndata import AnnData
import umap
class MinHashClustering():
    def __init__(self, minHashObject, clusteringObject):
        self._minHashObject = minHashObject
        self._clusteringObject = clusteringObject
        self._precomputed_graph = None
    def fit(self, X, y=None, pSaveMemory=None, pPca=None, pPcaDimensions=None, pUmap=None, pUmapDict=None):
        print('pUmapDict {}'.format(pUmapDict))
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

        if pPca:
            pca = PCA(n_components = min(self._precomputed_graph.shape) - 1)
            self._precomputed_graph = pca.fit_transform(self._precomputed_graph.todense())
            if pPcaDimensions:
                pPcaDimensions = min(pPcaDimensions, self._precomputed_graph.shape[0])
                self._precomputed_graph = self._precomputed_graph[:, :pPcaDimensions]
        if pUmap:
            
            reducer = umap.UMAP(n_neighbors=pUmapDict['umap_n_neighbors'], n_components=pUmapDict['umap_n_components'], metric=pUmapDict['umap_metric'], 
                                n_epochs=pUmapDict['umap_n_epochs'], 
                                learning_rate=pUmapDict['umap_learning_rate'], init=pUmapDict['umap_init'], min_dist=pUmapDict['umap_min_dist'], spread=pUmapDict['umap_spread'], 
                                set_op_mix_ratio=pUmapDict['umap_set_op_mix_ratio'], local_connectivity=pUmapDict['umap_local_connectivity'], 
                                repulsion_strength=pUmapDict['umap_repulsion_strength'], negative_sample_rate=pUmapDict['umap_negative_sample_rate'], transform_queue_size=pUmapDict['umap_transform_queue_size'], 
                                a=pUmapDict['umap_a'], b=pUmapDict['umap_b'], angular_rp_forest=pUmapDict['umap_angular_rp_forest'], 
                                target_n_neighbors=pUmapDict['umap_target_n_neighbors'], target_metric=pUmapDict['umap_target_metric'], 
                                target_weight=pUmapDict['umap_target_weight'],
                                force_approximation_algorithm=pUmapDict['umap_force_approximation_algorithm'], verbose=pUmapDict['umap_verbose'], unique=pUmapDict['umap_unique'])
            self._precomputed_graph = reducer.fit_transform(self._precomputed_graph)
        if pPca or pUmap:
            self._clusteringObject.fit(self._precomputed_graph)
            return 
        try:
            self._clusteringObject.fit(self._precomputed_graph)
        except:
            self._clusteringObject.fit(self._precomputed_graph.todense())
        return
	
    def fit_predict(self, X, y=None, pSaveMemory=None, pPca=None, pPcaDimensions=None, pUmap=None, **pUmapDict):

        self.fit(X, y, pSaveMemory=pSaveMemory, pPca=pPca, pPcaDimensions=pPcaDimensions)

        return self.predict(self._precomputed_graph, y, pPca=pPca, pPcaDimensions=pPcaDimensions, pUmap=pUmap, pUmapDict=pUmapDict)
		
    def predict(self, X, y=None, pPca=None, pPcaDimensions=None):
        if hasattr(self._clusteringObject, 'labels_'):
            return self._clusteringObject.labels_.astype(np.int)
        else:
            if pPca:
                if pPcaDimensions:
                    pPcaDimensions = min(pPcaDimensions, self._precomputed_graph.shape[0])
                    return self._clusteringObject.fit(self._precomputed_graph[:, :pPcaDimensions])
 
            return self._clusteringObject.predict(X)