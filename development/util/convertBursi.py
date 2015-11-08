#! /usr/bin/python
# Copyright 2015 Joachim Wolff
# Master Thesis
# Tutors: Fabrizio Costa
# Winter semester 2015/2016
#
# Chair of Bioinformatics
# Department of Computer Science
# Faculty of Engineering
# Albert-Ludwig-University Freiburg im Breisgau
#

import _convertDataToCpp
from eden.converter.graph.gspan import gspan_to_eden
from eden.graph import Vectorizer
from scipy.sparse import csr_matrix

if __name__ == "__main__":

    graphs = gspan_to_eden( 'http://www.bioinf.uni-freiburg.de/~costa/bursi.gspan' )
    vectorizer = Vectorizer( r=2,d=5 )
    datasetBursi = vectorizer.transform( graphs )
    X = csr_matrix(datasetBursi)
    instances, features = X.nonzero()
    data = X.data

    maxFeatures = 0
    countFeatures = 0;
    oldInstance = 0
    for i in instances:
        if (oldInstance == i):
            countFeatures += 1
        else:
            if countFeatures > maxFeatures:
                maxFeatures = countFeatures
            countFeatures = 0
        oldInstance = i

    _convertDataToCpp.parseAndStoreVector(instances.tolist(), features.tolist(), data.tolist(), X.shape[0], maxFeatures, "bursi")