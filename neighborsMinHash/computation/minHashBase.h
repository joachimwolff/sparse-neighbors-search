/**
 Copyright 2015 Joachim Wolff
 Master Project
 Tutors: Milad Miladi, Fabrizio Costa
 Summer semester 2015

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/

#include <Python.h>

// #include "typeDefinitions.h"

#ifndef INVERSE_INDEX
#define INVERSE_INDEX
#include "inverseIndex.h"
#endif
// #ifndef TYPE_DEFINTIONS
// #define TYPE_DEFINTIONS
// #include "typeDefinitions.h"
// #endif

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
    public:

  	MinHashBase(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions, int pFast);

  	~MinHashBase();
    // Calculate the inverse index for the given instances.
    void fit(umapVector* instanceFeatureVector); 
    // Extend the inverse index with the given instances.
    void partialFit(); 
    // Calculate k-nearest neighbors.
    neighborhood kneighbors(rawData pRawData, size_t pNneighbors, int pFast); 
    // Calculate k-nearest neighbors as a graph.

    void set_mOriginalData(SparseMatrixFloat* pOriginalData) {
      mOriginalData = pOriginalData;
      return;
    }
    size_t getNneighbors() { return mNneighbors; };
};