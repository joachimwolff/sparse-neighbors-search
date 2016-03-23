/**
 Copyright 2016 Joachim Wolff
 Master Thesis
 Tutors: Milad Miladi, Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwigs-University Freiburg im Breisgau
**/


#include "inverseIndex.h"
#include "hash.h"

#ifdef CUDA
#include "nearestNeighborsCuda.h"
#endif

#ifndef NEAREST_NEIGHBORS_H
#define NEAREST_NEIGHBORS_H

class NearestNeighbors {
  protected:
    InverseIndex* mInverseIndex;
    SparseMatrixFloat* mOriginalData;

	neighborhood computeNeighborhood();
    neighborhood computeExactNeighborhood();
  	neighborhood computeNeighborhoodGraph();

    size_t mNneighbors;
    int mFast;
    size_t mNumberOfCores;
    size_t mChunkSize;
    size_t mSimilarity;
    size_t mExcessFactor;
    float mCpuGpuLoadBalancing;
    Hash* mHash;
    #ifdef CUDA
    NearestNeighborsCuda* mNearestNeighborsCuda;
    #endif
    public:
    NearestNeighbors();

  	NearestNeighbors(size_t pNumberOfHashFunctions, size_t pShingleSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions, 
                    int pFast, int pSimilarity,
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance, 
                    int pRemoveHashFunctionWithLessEntriesAs, 
                    size_t pHashAlgorithm, size_t pBlockSize,
                    size_t pShingle, size_t pRemoveValueWithLeastSigificantBit,
                    float pCpuGpuLoadBalancing, size_t pRangeK_Wta);

  	~NearestNeighbors(); 
    // Calculate the inverse index for the given instances.
    void fit(const SparseMatrixFloat* pRawData); 
    // Extend the inverse index with the given instances.
    void partialFit(const SparseMatrixFloat* pRawData, size_t pStartIndex); 
    // Calculate k-nearest neighbors.
    neighborhood* kneighbors(const SparseMatrixFloat* pRawData, size_t pNneighbors, int pFast, int pSimilarity = -1); 

    void set_mOriginalData(SparseMatrixFloat* pOriginalData) {
      mOriginalData = pOriginalData;
      return;
    }
    SparseMatrixFloat* getOriginalData() {
        return mOriginalData;
    }
    size_t getNneighbors() { return mNneighbors; };
    
    distributionInverseIndex* getDistributionOfInverseIndex();
    
};
#endif // NEAREST_NEIGHBORS_H
