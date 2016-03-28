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
    int* mDev_JumpLength;
    float* mDev_DotProduct;
    int mNumberOfHashFunctions;
    int mShingleSize;
    int mShingle;
    int mHashAlgorithm;
    int mBlockSize;
    bool mDataOnGpu;
  public:
  	InverseIndexCuda(int pNumberOfHashFunctions, int pShingle, 
                        int pShingleSize, int pBlockSize, int pHashAlgorithm);
    ~InverseIndexCuda();
    void copyDataToGpu(SparseMatrixFloat* pRawData, int** pDevFeatureList,
                                      float** pDevValueList, int** pSizeList, int** pJumpList);
  	void computeSignaturesFittingOnGpu(SparseMatrixFloat* pRawData, 
                                    int pStartIndex, int pEndIndex, 
                                    int pNumberOfInstances, int pNumberOfBlocks, 
                                    int pNumberOfThreads, int pShingleFactor, 
                                    int pBlockSizeShingle,
                                    vvsize_t_p* pSignatures, int pRangeK);
    // void copyFittingDataToGpu(SparseMatrixFloat* pRawData, size_t pStartIndex);
    
   void computeSignaturesQueryOnGpu(SparseMatrixFloat* pRawData, 
                                                int pStartIndex, int pEndIndex, 
                                                int pNumberOfInstances, int pNumberOfBlocks, 
                                                int pNumberOfThreads, int pShingleFactor, 
                                                int pBlockSizeShingle,
                                                vvsize_t_p* pSignatures, int pRangeK);
   int** get_mDev_FeatureList() {
       return &mDev_FeatureList;
   };
   int** get_mDev_SizeOfInstanceList() {
       return &mDev_SizeOfInstanceList;
   };
   int** get_mDev_ComputedSignaturesPerInstance() {
       return &mDev_ComputedSignaturesPerInstance;
   };
   float** get_mDev_ValuesList() {
       return &mDev_ValuesList;
   };
   int** get_mDev_JumpLength() {
       return &mDev_JumpLength;
   };
   float** get_mDev_DotProduct() {
       return &mDev_DotProduct;
   };
};
#endif // INVERSE_INDEX_CUDA_H 