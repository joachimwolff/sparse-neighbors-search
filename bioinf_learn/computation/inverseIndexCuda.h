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
    int* mDev_FeatureList;
    int* mDev_SizeOfInstanceList;
    int* mDev_ComputedSignaturesPerInstance;
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
  	void computeSignaturesFittingOnGpu(const SparseMatrixFloat* pRawData, 
                                    size_t pStartIndex, size_t pEndIndex, 
                                    size_t pNumberOfInstances, size_t pNumberOfBlocks, 
                                    size_t pNumberOfThreads, size_t pShingleFactor, 
                                    size_t pBlockSizeShingle,
                                    vvsize_t_p* pSignatures);
    void copyFittingDataToGpu(const SparseMatrixFloat* pRawData);
    
   void computeSignaturesQueryOnGpu(const SparseMatrixFloat* pRawData, 
                                                size_t pStartIndex, size_t pEndIndex, 
                                                size_t pNumberOfInstances, size_t pNumberOfBlocks, 
                                                size_t pNumberOfThreads, size_t pShingleFactor, 
                                                size_t pBlockSizeShingle,
                                                vvsize_t_p* pSignatures);
   int* get_mDev_FeatureList() {
       return mDev_FeatureList;
   };
   int* get_mDev_SizeOfInstanceList() {
       return mDev_SizeOfInstanceList;
   };
   int* get_mDev_ComputedSignaturesPerInstance() {
       return mDev_ComputedSignaturesPerInstance;
   };
   float* get_mDev_ValuesList() {
       return mDev_ValuesList;
   };
};
#endif // INVERSE_INDEX_CUDA_H 