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

#include <Python.h>

#include "inverseIndex.h"


#ifndef MIN_HASH_BASE_H
#define MIN_HASH_BASE_H

class MinHashBase {
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

    public:

  	MinHashBase(size_t pNumberOfHashFunctions, size_t pShingleSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions, 
                    int pFast, int pSimilarity, size_t pBloomierFilter,
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance, 
                    int pRemoveHashFunctionWithLessEntriesAs,
                    size_t pHashAlgorithm, size_t pBlockSize,
                    size_t pShingle);

  	~MinHashBase();
    // Calculate the inverse index for the given instances.
    void fit(const SparseMatrixFloat* pRawData); 
    // Extend the inverse index with the given instances.
    void partialFit(); 
    // Calculate k-nearest neighbors.
    neighborhood* kneighbors(const SparseMatrixFloat* pRawData, size_t pNneighbors, int pFast, int pSimilarity = -1); 
    // Calculate k-nearest neighbors as a graph.

    void set_mOriginalData(SparseMatrixFloat* pOriginalData) {
      mOriginalData = pOriginalData;
      return;
    }
    size_t getNneighbors() { return mNneighbors; };
    
    distributionInverseIndex* getDistributionOfInverseIndex();
    
};
#endif // MIN_HASH_BASE_H
