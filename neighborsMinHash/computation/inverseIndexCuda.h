/**
 Copyright 2016 Joachim Wolff
 Master Thesis
 Tutor: Fabrizio Costa, Milad Miladi
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwigs-University Freiburg im Breisgau
**/

#include "typeDefinitions.h"
// #include "kernel.h"
#ifndef INVERSE_INDEX_CUDA_H
#define INVERSE_INDEX_CUDA_H
class InverseIndexCuda {

  private: 
    size_t* mDev_FeatureList;
    size_t* mDev_SizeOfInstanceList;
    size_t* mDev_ComputedSignaturesPerInstance;
    float* mDev_ValuesList;
    
    size_t mNumberOfHashFunctions;
    size_t mShingleSize;
    size_t mShingle;
    size_t mHashAlgorithm;
    size_t mBlockSize;
    
  public:
  	InverseIndexCuda(size_t pNumberOfHashFunctions, size_t pShingle, 
                        size_t pShingleSize, size_t pBlockSize);
    ~InverseIndexCuda();
  	vvsize_t_p* computeSignaturesOnGpu(const SparseMatrixFloat* pRawData, 
                                        size_t pStartIndex, size_t pEndIndex, 
                                        size_t pNumberOfInstances,
    size_t pNumberOfBlocks, size_t pNumberOfThreads);
    void copyDataToGpu(const SparseMatrixFloat* pRawData);
};
#endif // INVERSE_INDEX_CUDA_H