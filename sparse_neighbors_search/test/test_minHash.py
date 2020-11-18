import warnings
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=PendingDeprecationWarning)
import pytest
import os
from tempfile import NamedTemporaryFile, mkdtemp

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/")

from sparse_neighbors_search import MinHash
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
def test_minHash():

    path = ROOT + 'nagano_1mb_intermediate.npz'
    neighborhood_matrix = load_npz(path)
    minHash_object = MinHash(n_neighbors=500, number_of_hash_functions=20, number_of_cores=4,
                                     shingle_size=5, fast=True, maxFeatures=int(max(neighborhood_matrix.getnnz(1))), absolute_numbers=False, max_bin_size=100000,
                                     minimal_blocks_in_common=400, excess_factor=1, prune_inverse_index=False)

    minHash_object.fit(neighborhood_matrix[0:10, :])

    knn_graph = minHash_object.kneighbors_graph(mode='distance')