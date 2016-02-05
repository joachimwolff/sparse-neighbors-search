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

#include "inverseIndexCuda.h"
// #include "kSizeSortedMap.h"


// class sort_map {
//   public:
//     size_t key;
//     size_t val;
// };

// bool mapSortDescByValue(const sort_map& a, const sort_map& b) {
//         return a.val > b.val;
// };

InverseIndexCuda::InverseIndexCuda(size_t pNumberOfHashFunctions, size_t pShingleSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions, size_t pBloomierFilter,
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance, int pRemoveHashFunctionWithLessEntriesAs,
                    size_t pHashAlgorithm, size_t pBlockSize, size_t pShingle, size_t pRemoveValueWithLeastSigificantBit) {   
        // std::cout << __LINE__ << std::endl;
                        
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
    // mSignatureStorage = new umap_uniqueElement();
    // mHash = new Hash();
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
    
    // if (pBloomierFilter) {
    //     mInverseIndexStorage = new InverseIndexStorageBloomierFilter(inverseIndexSize, mMaxBinSize, maximalFeatures);
    // } else {
    //     mInverseIndexStorage = new InverseIndexStorageUnorderedMap(inverseIndexSize, mMaxBinSize);
    // }
    // mRemoveValueWithLeastSigificantBit = pRemoveValueWithLeastSigificantBit;
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
    cudaGetDevice(&whichDevice);
    cudaGetDeviceProperties(&prop, whichDevice);
    // if (pRawData->size() > maxBlocks) {
    //     for (size_t i = 0; i < pRawData->size(); ++i) {
    //         if (mNumberOfHashFunctions > prop.maxThreadsPerBlock) {
    //             for (size_t j = 0; j < mNumberOfHashFunctions; ++j) {
                    
    //             }
    //         }
            
    //     }
    // } else {
    //     if (mNumberOfHashFunctions > prop.maxThreadsPerBlock) {
       
    //     } else {
   
    size_t* result;
    result = (size_t*) malloc(pRawData->size() * mNumberOfHashFunctions * sizeof(size_t));
       
    size_t* dev_featureList;
    size_t* dev_sizeOfInstanceList;
    size_t* dev_computedSignaturesPerInstance;
    
    cudaHostAlloc((void **) &dev_featureList,
               pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(size_t),
               cudaHostAllocWriteCombined |
               cudaHostAllocMapped);
    cudaHostAlloc((void **) &dev_sizeOfInstanceList,
               pRawData->getNumberOfInstances() * sizeof(size_t),
               cudaHostAllocWriteCombined |
               cudaHostAllocMapped);
    cudaHostAlloc((void **) &dev_computedSignaturesPerInstance,
               pRawData->size() * mNumberOfHashFunctions * sizeof(size_t),
               cudaHostAllocMapped);
    // cudaMemcpy(dev_featureList, pRawData->getSparseMatrixIndex(),
    //             pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(size_t),
    //            cudaHostAllocWriteCombined |
    //            cudaHostAllocMapped);
    // cudaMemcpy(dev_sizeOfInstanceList, pRawData->getSparseMatrixSizeOfInstances(),
    //         pRawData->getNumberOfInstances() * sizeof(size_t),
    //         cudaHostAllocMapped);
    cudaHostGetDevicePointer(&dev_featureList, pRawData->getSparseMatrixIndex(), 0);
    cudaHostGetDevicePointer(&dev_sizeOfInstanceList, pRawData->getSparseMatrixSizeOfInstances(), 0);
    cudaHostGetDevicePointer(&dev_computedSignaturesPerInstance, result, 0);
    
    fitGpu<<<pRawData->getNumberOfInstances(), mNumberOfHashFunctions, mNumberOfHashFunctions>>>
    (dev_featureList, 
    dev_sizeOfInstanceList, 
    mNumberOfHashFunctions, 
    pRawData->getMaxNnz(),
            dev_computedSignaturesPerInstance, 
            pRawData->getNumberOfInstances());
    cudaThreadSynchronize();
    
    std::vector<std::vector<size_t> > hashFunctionHashValues(mNumberOfHashFunctions, std::vector<size_t>(0));
    std::vector<std::vector<vector<size_t> > > associatedIndexValues(mNumberOfHashFunctions, std::vector<size_t>(0));
    for (size_t i = 0; i < pRawData->getNumberOfInstances(); ++i) {
        // std::cout << result[i] ", ";
        for (size_t j = 0; j < mNumberOfHashFunctions; ++j) {
            hashFunctionHashValue[j].push_back(result[pRawData->getNumberOfInstances() * i + j]);
            associatedIndexValues[j].push_back(i);
        }
    }
    
    cudaFree(dev_featureList);
    cudaFree(dev_sizeOfInstanceList);
    cudaFree(dev_computedSignaturesPerInstance);
    // cudaMemcpy(result, dev_computedSignaturesPerInstance,
    //            pRawData->size() * mNumberOfHashFunctions * sizeof(size_t),
    //            cudaMemcpyDeviceToHost );
    
        // }
        
    // }
    // put values in vectors, sort them.
    // copy index to gpu
    
    
}

neighborhood* InverseIndexCuda::kneighbors(const umap_uniqueElement* pSignaturesMap, 
                                        const size_t pNneighborhood, const bool pDoubleElementsStorageCount) {
// compute hits in the inverse index on the gpu and 
// return a list with all the associated index values per hash function
// process them on the gpu,
// compute exact neighbors based on the hits on gpu.

}