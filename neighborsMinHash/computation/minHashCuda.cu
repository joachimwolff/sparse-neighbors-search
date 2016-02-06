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

#include "minHashCuda.h"





MinHashCuda::MinHashCuda(size_t pNumberOfHashFunctions, size_t pShingleSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions, 
                    int pFast, int pSimilarity, size_t pBloomierFilter,
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance, 
                    int pRemoveHashFunctionWithLessEntriesAs, size_t pHashAlgorithm,
                    size_t pBlockSize, size_t pShingle,
                    size_t pRemoveValueWithLeastSigificantBit):MinHashBase() {
            printf("%i\n", __LINE__);
       
       mInverseIndex = new InverseIndexCuda(pNumberOfHashFunctions, pShingleSize,
                                            pNumberOfCores, pChunkSize,
                                            pMaxBinSize, pMinimalBlocksInCommon, 
                                            pExcessFactor, pMaximalNumberOfHashCollisions, pBloomierFilter,
                                            pPruneInverseIndex, pPruneInverseIndexAfterInstance, 
                                            pRemoveHashFunctionWithLessEntriesAs, pHashAlgorithm, pBlockSize, pShingle,
                                            pRemoveValueWithLeastSigificantBit);

        mNneighbors = pSizeOfNeighborhood;
        mFast = pFast;
        mNumberOfCores = pNumberOfCores;
        mChunkSize = pChunkSize;
        mSimilarity = pSimilarity;
            printf("%i\n", __LINE__);
        
}

MinHashCuda::~MinHashCuda() {
    delete mInverseIndex;
}
void MinHashCuda::fit(const SparseMatrixFloat* pRawData) {
            printf("%i\n", __LINE__);

    mInverseIndex->fit(pRawData);
            printf("%i\n", __LINE__);

}
neighborhood MinHashCuda::radiusNeighbors() {
    neighborhood foo;
    return foo;
}
neighborhood MinHashCuda::radiusNeighborsGraph() {
    neighborhood foo;
    return foo;
}
neighborhood MinHashCuda::fitRadiusNeighbors() {
    neighborhood foo;
    return foo;
}
neighborhood MinHashCuda::fitRadiusNeighborsGraph() {
    neighborhood foo;
    return foo;
}