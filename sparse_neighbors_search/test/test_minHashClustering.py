import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)
import pytest
import os
from tempfile import NamedTemporaryFile, mkdtemp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/")

from sparse_neighbors_search import MinHash
from sparse_neighbors_search import MinHashClustering
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz

from sklearn.cluster import SpectralClustering


def test_minHashClustering():

    path = ROOT + 'nagano_1mb_intermediate.npz'
    neighborhood_matrix = load_npz(path)

    minHash_object = MinHash(n_neighbors=500, number_of_hash_functions=20, number_of_cores=4,
                             shingle_size=5, fast=True, maxFeatures=int(max(neighborhood_matrix.getnnz(1))), absolute_numbers=False, max_bin_size=100000,
                             minimal_blocks_in_common=400, excess_factor=1, prune_inverse_index=False)
    cluster_object = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_jobs=2, random_state=0)
    minHashClustering = MinHashClustering(minHashObject=minHash_object, clusteringObject=cluster_object)
    minHashClustering.fit(X=neighborhood_matrix, pSaveMemory=0.1, pPca=False, pPcaDimensions=None, pUmap=False, pUmapDict=None)

    labels_clustering = minHashClustering.predict(minHashClustering._precomputed_graph)


def test_minHashClustering_pca():

    path = ROOT + 'nagano_1mb_intermediate.npz'
    neighborhood_matrix = load_npz(path)

    minHash_object = MinHash(n_neighbors=500, number_of_hash_functions=20, number_of_cores=4,
                             shingle_size=5, fast=True, maxFeatures=int(max(neighborhood_matrix.getnnz(1))), absolute_numbers=False, max_bin_size=100000,
                             minimal_blocks_in_common=400, excess_factor=1, prune_inverse_index=False)
    cluster_object = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_jobs=2, random_state=0)
    minHashClustering = MinHashClustering(minHashObject=minHash_object, clusteringObject=cluster_object)
    minHashClustering.fit(X=neighborhood_matrix, pSaveMemory=0.1, pPca=True, pPcaDimensions=2, pUmap=False, pUmapDict=None)

    labels_clustering = minHashClustering.predict(minHashClustering._precomputed_graph)

def test_minHashClustering_umap():

    path = ROOT + 'nagano_1mb_intermediate.npz'
    neighborhood_matrix = load_npz(path)

    minHash_object = MinHash(n_neighbors=500, number_of_hash_functions=20, number_of_cores=4,
                             shingle_size=5, fast=True, maxFeatures=int(max(neighborhood_matrix.getnnz(1))), absolute_numbers=False, max_bin_size=100000,
                             minimal_blocks_in_common=400, excess_factor=1, prune_inverse_index=False)
    cluster_object = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_jobs=2, random_state=0)
    minHashClustering = MinHashClustering(minHashObject=minHash_object, clusteringObject=cluster_object)
    minHashClustering.fit(X=neighborhood_matrix, pSaveMemory=0.1, pPca=False, pPcaDimensions=None, pUmap=True, pUmapDict=None)

    labels_clustering = minHashClustering.predict(minHashClustering._precomputed_graph)

def test_minHashClustering_pca_umap():

    path = ROOT + 'nagano_1mb_intermediate.npz'
    neighborhood_matrix = load_npz(path)

    minHash_object = MinHash(n_neighbors=500, number_of_hash_functions=20, number_of_cores=4,
                             shingle_size=5, fast=True, maxFeatures=int(max(neighborhood_matrix.getnnz(1))), absolute_numbers=False, max_bin_size=100000,
                             minimal_blocks_in_common=400, excess_factor=1, prune_inverse_index=False)
    cluster_object = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_jobs=2, random_state=0)
    minHashClustering = MinHashClustering(minHashObject=minHash_object, clusteringObject=cluster_object)
    minHashClustering.fit(X=neighborhood_matrix, pSaveMemory=0.1, pPca=True, pPcaDimensions=10, pUmap=True, pUmapDict=None)

    labels_clustering = minHashClustering.predict(minHashClustering._precomputed_graph)

