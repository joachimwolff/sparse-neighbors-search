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
#include "hash.h"
// #include "inverseIndexStorage.h"
// #include "inverseIndexStorageBloomierFilter.h"
#include "inverseIndexStorageUnorderedMap.h"
#ifdef CUDA
#include "inverseIndexCuda.h"
#endif
#include "typeDefinitions.h"


#ifndef INVERSE_INDEX_H
#define INVERSE_INDEX_H
class InverseIndex {

  protected: 
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
    size_t mInverseIndexSize;
    size_t mMaxNnz;
    float mCpuGpuLoadBalancing;
    size_t mRangeK_Wta;
    size_t mGpuHash; 


    InverseIndexStorageUnorderedMap* mInverseIndexStorage;
  	umap_uniqueElement* mSignatureStorage;
    Hash* mHash;
    #ifdef CUDA
    InverseIndexCuda* mInverseIndexCuda;
    #endif
    vsize_t* shingle(vsize_t* pSignature);
  public:
    InverseIndex();

  	InverseIndex(size_t pNumberOfHashFunctions, size_t pShingleSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions,
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance,
                    int pRemoveHashFunctionWithLessEntriesAs, size_t pHashAlgorithm, 
                    size_t pBlockSize, size_t pShingle, size_t pRemoveValueWithLeastSigificantBit,
                    float pCpuGpuLoadBalancing, size_t pGpuHash, size_t pRangeK_Wta);
    ~InverseIndex();
  	vsize_t* computeSignature(SparseMatrixFloat* pRawData, const size_t pInstance);
  	vsize_t* computeSignatureWTA(SparseMatrixFloat* pRawData, const size_t pInstance);
    vvsize_t_p* computeSignatureVectors(SparseMatrixFloat* pRawData, const bool pFitting);
  	umap_uniqueElement* computeSignatureMap(SparseMatrixFloat* pRawData);
  	void fit(SparseMatrixFloat* pRawData, size_t pStartIndex=0);
  	neighborhood* kneighbors(const umap_uniqueElement* pSignaturesMap, 
                                const size_t pNneighborhood, 
                                const bool pDoubleElementsStorageCount,
                                const bool pNoneSingleInstance=true);
  	umap_uniqueElement* getSignatureStorage() { 
      return mSignatureStorage;
    };
    distributionInverseIndex* getDistribution();
    vvsize_t_p* computeSignaturesOnGpu(SparseMatrixFloat* pRawData, size_t pStartIndex, size_t pEndIndex, size_t pNumberOfInstances,
    size_t pNumberOfBlocks, size_t pNumberOfThreads);
                
};
#endif // INVERSE_INDEX_H