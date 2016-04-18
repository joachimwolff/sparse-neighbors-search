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
    size_t* mDev_SizeOfInstanceList;
    int* mDev_ComputedSignaturesPerInstance;
    float* mDev_ValuesList;
    size_t* mDev_JumpLength;
    float* mDev_DotProduct;
    
    size_t mNumberOfHashFunctions;
    size_t mShingleSize;
    size_t mShingle;
    size_t mHashAlgorithm;
    size_t mBlockSize;
    bool mDataOnGpu;
  public:
  	InverseIndexCuda(size_t pNumberOfHashFunctions, size_t pShingle, 
                        size_t pShingleSize, size_t pBlockSize, size_t pHashAlgorithm);
    ~InverseIndexCuda();
    void copyDataToGpu(SparseMatrixFloat* pRawData, int** pDevFeatureList,
                                      float** pDevValueList, size_t** pSizeList);
  	void computeSignaturesFittingOnGpu(SparseMatrixFloat* pRawData, 
                                        size_t pStartIndex, size_t pEndIndex, 
                                        size_t pNumberOfInstances, size_t pNumberOfBlocks, 
                                        size_t pNumberOfThreads, size_t pShingleFactor, 
                                        size_t pBlockSizeShingle,
                                        vvsize_t_p* pSignatures, size_t pRangeK);
    // void copyFittingDataToGpu(SparseMatrixFloat* pRawData, size_t pStartIndex);
    
   void computeSignaturesQueryOnGpu(SparseMatrixFloat* pRawData, 
                                                size_t pStartIndex, size_t pEndIndex, 
                                                size_t pNumberOfInstances, size_t pNumberOfBlocks, 
                                                size_t pNumberOfThreads, size_t pShingleFactor, 
                                                size_t pBlockSizeShingle,
                                                vvsize_t_p* pSignatures, size_t pRangeK);
   int** get_mDev_FeatureList() {
       return &mDev_FeatureList;
   };
   size_t** get_mDev_SizeOfInstanceList() {
       return &mDev_SizeOfInstanceList;
   };
   int** get_mDev_ComputedSignaturesPerInstance() {
       return &mDev_ComputedSignaturesPerInstance;
   };
   float** get_mDev_ValuesList() {
       return &mDev_ValuesList;
   };
   size_t** get_mDev_JumpLength() {
       return &mDev_JumpLength;
   };
   float** get_mDev_DotProduct() {
       return &mDev_DotProduct;
   };
};
#endif // INVERSE_INDEX_CUDA_H 