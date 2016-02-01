/**
 Copyright 2015 Joachim Wolff
 Master Thesis
 Tutors: Fabrizio Costa, Milad Miladi
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/

#include <algorithm>
#include <iostream>
#include <iterator>
#include <utility>

#ifdef OPENMP
#include <omp.h>
#endif

#include "inverseIndex.h"
#include "kSizeSortedMap.h"


class sort_map {
  public:
    size_t key;
    size_t val;
};

bool mapSortDescByValue(const sort_map& a, const sort_map& b) {
        return a.val > b.val;
};

InverseIndexCuda::InverseIndexCuda(size_t pNumberOfHashFunctions, size_t pShingleSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions, size_t pBloomierFilter,
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance, int pRemoveHashFunctionWithLessEntriesAs,
                    size_t pHashAlgorithm, size_t pBlockSize, size_t pShingle, size_t pRemoveValueWithLeastSigificantBit) {   
        std::cout << __LINE__ << std::endl;
                        
    mNumberOfHashFunctions = pNumberOfHashFunctions;
    mShingleSize = pShingleSize;
    mNumberOfCores = pNumberOfCores;
    mChunkSize = pChunkSize;
    mMaxBinSize = pMaxBinSize;
    mMinimalBlocksInCommon = pMinimalBlocksInCommon;
    mExcessFactor = pExcessFactor;
    mMaximalNumberOfHashCollisions = pMaximalNumberOfHashCollisions;
    mPruneInverseIndex = pPruneInverseIndex;
    mPruneInverseIndexAfterInstance = pPruneInverseIndexAfterInstance;
    mRemoveHashFunctionWithLessEntriesAs = pRemoveHashFunctionWithLessEntriesAs;
    mHashAlgorithm = pHashAlgorithm;
    mSignatureStorage = new umap_uniqueElement();
    mHash = new Hash();
    mBlockSize = pBlockSize;
    mShingle = pShingle;
    size_t maximalFeatures = 5000;
    size_t inverseIndexSize;
    if (mShingle == 0) {
        if (mBlockSize == 0) {
            mBlockSize = 1;
        }
        inverseIndexSize = mNumberOfHashFunctions * mBlockSize;
    } else {
        inverseIndexSize = ceil(((float) (mNumberOfHashFunctions * mBlockSize) / (float) mShingleSize));        
    }
    if (pBloomierFilter) {
        mInverseIndexStorage = new InverseIndexStorageBloomierFilter(inverseIndexSize, mMaxBinSize, maximalFeatures);
    } else {
        mInverseIndexStorage = new InverseIndexStorageUnorderedMap(inverseIndexSize, mMaxBinSize);
    }
    mRemoveValueWithLeastSigificantBit = pRemoveValueWithLeastSigificantBit;
        // std::cout << __LINE__ << std::endl;
    
}
 
InverseIndexCuda::~InverseIndexCuda() {
    delete mSignatureStorage;
    delete mHash;
    delete mInverseIndexStorage;
}

distributionInverseIndex* InverseIndexCuda::getDistribution() {
    return mInverseIndexStorage->getDistribution();
}

 // compute the signature for one instance
vsize_t InverseIndexCuda::computeSignature(const SparseMatrixFloat* pRawData, const size_t pInstance) {

   
}

vsize_t InverseIndexCuda::shingle(vsize_t pSignature) {
  

}

vsize_t InverseIndexCuda::computeSignatureWTA(const SparseMatrixFloat* pRawData, const size_t pInstance) {
    
          
}

umap_uniqueElement* InverseIndexCuda::computeSignatureMap(const SparseMatrixFloat* pRawData) {
}
void InverseIndexCuda::fit(const SparseMatrixFloat* pRawData) {
    int maxBlocks = 65535;
    cudaDeviceProp prop;
    int whichDevice;
    cudeGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice)
    if (pRawData->size() > maxBlocks) {
        for (size_t i = 0; i < pRawData->size(); ++i) {
            if (mNumberOfHashFunctions > prop.maxThreadsPerBlock){
                for (size_t j = 0; j < mNumberOfHashFunctions; ++j) {
                    
                }
            }
            
        }
    } else {
        if (mNumberOfHashFunctions > prop.maxThreadsPerBlock) {
       
        } else {
            fit<<<pRawData->size(), mNumberOfHashFunctions, 0, stream>();
        }
        
    }
}

neighborhood* InverseIndexCuda::kneighbors(const umap_uniqueElement* pSignaturesMap, 
                                        const size_t pNneighborhood, const bool pDoubleElementsStorageCount) {

}