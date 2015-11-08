#! /usr/bin/python
import time
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_sparse_uncorrelated
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LSHForest
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import neighborsMinHash as kneighbors
from neighborsMinHash.util import create_dataset
from neighborsMinHash.util import create_dataset_fixed_nonzero
from neighborsMinHash.util import measure_performance
from neighborsMinHash.util import plotData


import random

from scipy.sparse import dok_matrix
from scipy.sparse import rand
from scipy.sparse import vstack

from sklearn.random_projection import SparseRandomProjection

# import pyflann
import annoy

from eden.converter.graph.gspan import gspan_to_eden
from eden.graph import Vectorizer
graphs = gspan_to_eden( 'http://www.bioinf.uni-freiburg.de/~costa/bursi.gspan' )
vectorizer = Vectorizer( r=2,d=5 )
datasetBursi = vectorizer.transform( graphs )

dimensions = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
projected_data = []
for d in dimensions: 
    data_projection = SparseRandomProjection(n_components=d, random_state=1)
    projected_data.append(data_projection.fit_transform(datasetBursi))

n_neighbors_sklearn = 5
n_neighbors_minHash = 5
returnValuesBursi = measure_performance([datasetBursi]*9, n_neighbors_sklearn, n_neighbors_minHash, number_of_hashfunctions=20000, dataset_dense=projected_data)