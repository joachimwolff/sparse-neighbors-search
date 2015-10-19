/**
 Copyright 2015 Joachim Wolff
 Master Thesis
 Tutors: Milad Miladi, Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/

#include <iostream>
#include <iterator>
#include <algorithm>
#include <utility>

#ifdef OPENMP
#include <omp.h>
#endif

#include "minHash.h"




MinHash::MinHash(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions):MinHashBase(pNumberOfHashFunctions, pBlockSize,
                                                pNumberOfCores, pChunkSize, pMaxBinSize,
                                                pSizeOfNeighborhood, pMinimalBlocksInCommon,
                                                pExcessFactor, pMaximalNumberOfHashCollisions) {
    
   
    // inverseIndex = new std::vector<umapVector >();
    // signatureStorage = new umap_pair_vector();
    // size_t inverseIndexSize = ceil(((float) numberOfHashFunctions / (float) blockSize)+1);
    // inverseIndex->resize(inverseIndexSize);
}


MinHash::~MinHash() {
}


