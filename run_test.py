#!/usr/bin/env python

from sparse_neighbors_search import MinHash
from sparse_neighbors_search import MinHashClassifier
from sparse_neighbors_search import WtaHash
from sparse_neighbors_search import WtaHashClassifier

import numpy as np 
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from eden.converter.graph.gspan import gspan_to_eden
from eden.graph import Vectorizer
import os.path
import time
import pickle
def load_bursi():
    if not os.path.isfile('/home/joachim/data/minhash/dataset_bursi'):
        graphs = None
        if not os.path.isfile('/home/joachim/data/minhash/bursi.gspan'):
            graphs = gspan_to_eden( 'http://www.bioinf.uni-freiburg.de/~costa/bursi.gspan' )
            pickle.dump(graphs, open('/home/joachim/data/minhash/bursi.gspan', 'wb'))
        else:
            graphs = pickle.load(open('/home/joachim/data/minhash/bursi.gspan', 'rb'))
        vectorizer = Vectorizer( r=2,d=5 )
        dataset = vectorizer.transform( graphs )
        pickle.dump(dataset, open('/home/joachim/data/minhash/dataset_bursi', 'wb'))
        return dataset
    else:
        return pickle.load(open('/home/joachim/data/minhash/dataset_bursi', 'rb'))
if __name__ == "__main__":
    dataset = load_bursi()
    start = time.time()
    sklearn = NearestNeighbors(n_neighbors=5, n_jobs=4)
    sklearn.fit(dataset)
    end = time.time()
    print "fitting: ", end - start
    start = time.time()
    neighbors = sklearn.kneighbors(return_distance=False)
    end = time.time()
    print neighbors
    print 'neighbors computing time: ', end - start 

    start = time.time()
    minhash = MinHash(n_neighbors=5)
    minhash.fit(dataset)
    end = time.time()
    print "fitting: ", end - start
    start = time.time()
    neighbors = minhash.kneighbors(return_distance=False)
    end = time.time()
    print neighbors
    print 'neighbors computing time: ', end - start 
    # n_neighbors_minHash = MinHash(n_neighbors = 4)
