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

__author__ = 'joachimwolff'

from .neighbors.minHashClassifier import MinHashClassifier
from .neighbors.minHash import MinHash
from .neighbors.wtaHashClassifier import WtaHashClassifier
from .neighbors.wtaHash import WtaHash
# import cluster
from .cluster.minHashSpectralClustering import MinHashSpectralClustering
from .cluster.minHashDBSCAN import MinHashDBSCAN
from .cluster.minHashClustering import MinHashClustering

import logging
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logging.getLogger('numba').setLevel(logging.ERROR)