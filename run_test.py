#!/usr/bin/env python

from sparse_neighbors_search import MinHash
from sparse_neighbors_search import MinHashClassifier
from sparse_neighbors_search import WtaHash
from sparse_neighbors_search import WtaHashClassifier

import numpy as np 
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

from eden.converter.graph.gspan import gspan_to_eden
from eden.graph import Vectorizer
import os.path
import time
import pickle
from scipy.io import mmwrite
from scipy.io import mmread

def load_bursi():
    # if not os.path.isfile('/home/wolffj/data/minhash/dataset_bursi'):
    #     graphs = None
    #     if not os.path.isfile('/home/wolffj/data/minhash/bursi.gspan'):
    #         graphs = gspan_to_eden( 'http://www.bioinf.uni-freiburg.de/~costa/bursi.gspan' )
    #         # pickle.dump(graphs, open('/home/wolffj/data/minhash/bursi.gspan', 'wb'))
    #     else:
    #         graphs = pickle.load(open('/home/wolffj/data/minhash/bursi.gspan', 'rb'))
    #     vectorizer = Vectorizer( r=2,d=5 )
    #     dataset = vectorizer.transform( graphs )
    #     pickle.dump(dataset, open('/home/wolffj/data/minhash/dataset_bursi', 'wb'))
    #     mmwrite(open("bursi.mtx", 'w+'), dataset)

    #     return dataset
    # else:
        # return pickle.load(open('/home/wolffj/data/minhash/dataset_bursi', 'rb'))
    return mmread(open("bursi.mtx", 'r'))
if __name__ == "__main__":
    dataset = load_bursi()

    # data = []
    # print dataset.get_shape()
    # for i in xrange(dataset.get_shape()[0]):
    #     if i < 100:
    #         data.append(dataset.getrow(i))
    # print data
    # dataset = np.vstack(data)
    # dataset = dataset.tocsr()[:100]
    # mmwrite(open("bursi_test.mtx", 'w+'), dataset)
    # print type(data)
    start = time.time()
    sklearn = NearestNeighbors(n_neighbors=5, n_jobs=8)
    sklearn.fit(dataset)
    end = time.time()
    print "fitting: ", end - start
    start = time.time()
    neighbors_sklearn = sklearn.kneighbors(return_distance=False)
    end = time.time()
    print neighbors_sklearn
    print 'neighbors computing time: ', end - start 

    start = time.time()
    minhash = MinHash(n_neighbors=5, number_of_cores=8)
    minhash.fit(dataset)
    end = time.time()
    print "fitting: ", end - start
    start = time.time()
    neighbors_minHash = minhash.kneighbors(return_distance=False)
    end = time.time()
    print neighbors_minHash
    print 'neighbors computing time: ', end - start 

    accuracy = 0;
    for i in xrange(len(neighbors_minHash)):
        accuracy += len(np.intersect1d(neighbors_minHash[i], neighbors_sklearn[i]))
    
    print "Accuracy: ", accuracy / float(len(neighbors_minHash) * len(neighbors_sklearn[0]))
    # n_neighbors_minHash = MinHash(n_neighbors = 4)
    # mmwrite(open("bursi_neighbors.mtx", 'w+'), neighbors)
    # mmwrite(open("bursi_values.mtx", 'w+'), dataset)