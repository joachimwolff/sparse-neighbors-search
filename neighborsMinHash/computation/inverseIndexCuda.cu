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
#include "kernel.h"


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
                    size_t pHashAlgorithm, size_t pBlockSize, size_t pShingle, size_t pRemoveValueWithLeastSigificantBit):InverseIndex() {   
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
    mBlockSize = pBlockSize;
    mShingle = pShingle;
    size_t inverseIndexSize;
    if (mShingle == 0) {
        if (mBlockSize == 0) {
            mBlockSize = 1;
        }
        inverseIndexSize = mNumberOfHashFunctions * mBlockSize;
    } else {
        inverseIndexSize = ceil(((float) (mNumberOfHashFunctions * mBlockSize) / (float) mShingleSize));        
    }
    
    
}
 
InverseIndexCuda::~InverseIndexCuda() {
   cudaFree(mDev_FeatureList);
   cudaFree(mDev_ComputedSignaturesPerInstance);
   cudaFree(mDev_SizeOfInstanceList);
   cudaFree(mDev_ValuesList);
  
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
    // printf("foo");
    // fflush(stdout);
    int maxBlocks = 65535;
    // printf("number of hash functions: %zu", mNumberOfHashFunctions);
    // fflush(stdout);
    if (mNumberOfHashFunctions % 32 != 0) {
        mNumberOfHashFunctions += 32 - (mNumberOfHashFunctions % 32);
    }
    // printf("number of hash functions: %zu", mNumberOfHashFunctions);
    // fflush(stdout);
    
    // memory for instances and their featureIds
    cudaMalloc((void **) &mDev_FeatureList,
               pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(size_t));
    // memory for the values of the features of the instances
    cudaMalloc((void **) &mDev_ValuesList, 
                pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(float));
    // memory for the number of features per instance
    cudaMalloc((void **) &mDev_SizeOfInstanceList,
               pRawData->getNumberOfInstances() * sizeof(size_t));
    // memory for the inverse index on the gpu.
    // for each instance the number of hash functions
    cudaMalloc((void **) &mDev_ComputedSignaturesPerInstance,
               pRawData->getNumberOfInstances()* mNumberOfHashFunctions * sizeof(size_t));
    // copy instances and their feature ids to the gpu
    cudaMemcpy(mDev_FeatureList, pRawData->getSparseMatrixIndex(),
                pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(size_t),
               cudaMemcpyHostToDevice);
    // copy instances and their values for each feature to the gpu
    cudaMemcpy(mDev_ValuesList, pRawData->getSparseMatrixValues(),
                pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(float),
               cudaMemcpyHostToDevice);
    // copy the size of all instances to the gpu               
    cudaMemcpy(mDev_SizeOfInstanceList, pRawData->getSparseMatrixSizeOfInstances(),
            pRawData->getNumberOfInstances() * sizeof(size_t),
            cudaMemcpyHostToDevice);
    // fitGpu<<<pRawData->getNumberOfInstances(), mNumberOfHashFunctions, mNumberOfHashFunctions>>>
    // compute values on the gpu
    fitCuda<<<32, 10, mNumberOfHashFunctions>>>
    (mDev_FeatureList, 
    mDev_SizeOfInstanceList,  
    mNumberOfHashFunctions, 
    pRawData->getMaxNnz(),
            mDev_ComputedSignaturesPerInstance, 
            pRawData->getNumberOfInstances());
    // copy values back for debugging
    size_t* instancesHashValues = (size_t*) malloc(pRawData->getNumberOfInstances() * mNumberOfHashFunctions * sizeof(size_t));
    cudaMemcpy(instancesHashValues, mDev_ComputedSignaturesPerInstance, 
                pRawData->getNumberOfInstances() * mNumberOfHashFunctions * sizeof(size_t),
                cudaMemcpyDeviceToHost);
   for(size_t i = 0; i < pRawData->getNumberOfInstances(); ++i) {
       printf("Instance: %zu of %zu: ", i, pRawData->getNumberOfInstances());
       for (size_t j = 0; j < mNumberOfHashFunctions; ++j) {
           printf("%zu,", instancesHashValues[i*mNumberOfHashFunctions + j]);
       }
       printf("\n");
   }
   free(instancesHashValues);
}

neighborhood* InverseIndexCuda::kneighbors(const umap_uniqueElement* pSignaturesMap, 
                                        const size_t pNneighborhood, const bool pDoubleElementsStorageCount) {
// compute hits in the inverse index on the gpu and 
// return a list with all the associated index values per hash function
// process them on the gpu,
// compute exact neighbors based on the hits on gpu.

}