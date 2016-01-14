/**
 Copyright 2015 Joachim Wolff
 Master Thesis
 Tutors: Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/

#ifdef OPENMP
#include <omp.h>
#endif

#include "minHash.h"




MinHash::MinHash(size_t pNumberOfHashFunctions, size_t pShingleSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions, 
                    int pFast, int pSimilarity, size_t pBloomierFilter,
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance, 
                    int pRemoveHashFunctionWithLessEntriesAs, size_t pHashAlgorithm,
                    size_t pBlockSize):MinHashBase(pNumberOfHashFunctions, pShingleSize,
                                                pNumberOfCores, pChunkSize, pMaxBinSize,
                                                pSizeOfNeighborhood, pMinimalBlocksInCommon,
                                                pExcessFactor, pMaximalNumberOfHashCollisions,
                                                pFast, pSimilarity, pBloomierFilter,
                                                pPruneInverseIndex, pPruneInverseIndexAfterInstance,
                                                pRemoveHashFunctionWithLessEntriesAs, pHashAlgorithm,
                                                pBlockSize) {
}

MinHash::~MinHash() {
}

