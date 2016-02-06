/**
 Copyright 2015 Joachim Wolff
 Master Thesis
 Tutor: Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/

#include <functional>
#include "inverseIndex.h"
// #include "hash.h"
// #include "inverseIndexStorage.h"
// #include "inverseIndexStorageBloomierFilter.h"
// #include "inverseIndexStorageUnorderedMap.h"

#include "typeDefinitions.h"

#ifndef INVERSE_INDEX_CUDA_H
#define INVERSE_INDEX_CUDA_H
class InverseIndexCuda : public InverseIndex {

  private: 
  	// const double A = sqrt(2) - 1;
      
    size_t* mDev_FeatureList;
    size_t* mDev_SizeOfInstanceList;
    size_t* mDev_ComputedSignaturesPerInstance;
      
    size_t mNumberOfHashFunctions;
    size_t mShingleSize;
    size_t mNumberOfCores;
    size_t mChunkSize;
    size_t mMaxBinSize;
    size_t mMinimalBlocksInCommon;
    size_t mExcessFactor;
    size_t mMaximalNumberOfHashCollisions;
    size_t mShingle;
    size_t mHashAlgorithm;
    size_t mDoubleElementsStorageCount = 0;
    size_t mDoubleElementsQueryCount = 0;
    int mPruneInverseIndex;
    float mPruneInverseIndexAfterInstance;
    int mRemoveHashFunctionWithLessEntriesAs;
    size_t mBlockSize;
    size_t mRemoveValueWithLeastSigificantBit;
    
    InverseIndexStorage* mInverseIndexStorage;
  	umap_uniqueElement* mSignatureStorage;
    Hash* mHash;
    vsize_t shingle(vsize_t pSignature);
  public:
  	InverseIndexCuda(size_t pNumberOfHashFunctions, size_t pShingleSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions, size_t pBloomierFilter,
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance,
                    int pRemoveHashFunctionWithLessEntriesAs, size_t pHashAlgorithm, 
                    size_t pBlockSize, size_t pShingle, size_t pRemoveValueWithLeastSigificantBit);
    ~InverseIndexCuda();
  	vsize_t computeSignature(const SparseMatrixFloat* pRawData, const size_t pInstance);
  	vsize_t computeSignatureWTA(const SparseMatrixFloat* pRawData, const size_t pInstance);
      
  	umap_uniqueElement* computeSignatureMap(const SparseMatrixFloat* pRawData);
  	void fit(const SparseMatrixFloat* pRawData);
  	neighborhood* kneighbors(const umap_uniqueElement* signaturesMap, const size_t pNneighborhood, const bool pDoubleElementsStorageCount);
  	umap_uniqueElement* getSignatureStorage() { 
      return mSignatureStorage;
    };
    distributionInverseIndex* getDistribution();
};
#endif // INVERSE_INDEX_CUDA_H