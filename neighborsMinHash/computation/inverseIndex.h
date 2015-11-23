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


#include "typeDefinitions.h"


#ifndef INVERSE_INDEX_H
#define INVERSE_INDEX_H
class InverseIndex {

  protected: 
  	const double A = sqrt(2) - 1;
    size_t mNumberOfHashFunctions;
    size_t mBlockSize;
    size_t mNumberOfCores;
    size_t mChunkSize;
    size_t mMaxBinSize;
    size_t mMinimalBlocksInCommon;
    size_t mExcessFactor;
    size_t mMaximalNumberOfHashCollisions;
    
    size_t mDoubleElementsStorageCount = 0;
    size_t mDoubleElementsQueryCount = 0;

  	umap_uniqueElement* mSignatureStorage;
  	  
    Hash* mHash;

  public:
  	InverseIndex(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions) {
          mNumberOfHashFunctions = pNumberOfHashFunctions;
          mBlockSize = pBlockSize;
          mNumberOfCores = pNumberOfCores;
          mChunkSize = pChunkSize;
          mMaxBinSize = pMaxBinSize;
          mMinimalBlocksInCommon = pMinimalBlocksInCommon;
          mExcessFactor = pExcessFactor;
          mMaximalNumberOfHashCollisions = pMaximalNumberOfHashCollisions;
          mSignatureStorage = new umap_uniqueElement();
          mHash = new Hash();
    };
  	virtual ~InverseIndex();
  	virtual vsize_t* computeSignature(const SparseMatrixFloat* pRawData, const size_t pInstance);
  	virtual umap_uniqueElement* computeSignatureMap(const SparseMatrixFloat* pRawData);
  	virtual void fit(const SparseMatrixFloat* pRawData);
  	virtual neighborhood* kneighbors(const umap_uniqueElement* signaturesMap, const int pNneighborhood, const bool pDoubleElementsStorageCount);
  	virtual umap_uniqueElement* getSignatureStorage() { 
      return mSignatureStorage;
    };
};
#endif // INVERSE_INDEX_H