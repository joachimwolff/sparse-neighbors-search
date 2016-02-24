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
  	void computeSignaturesOnGpu(const SparseMatrixFloat* pRawData, 
                                    size_t pStartIndex, size_t pEndIndex, 
                                    size_t pNumberOfInstances, size_t pNumberOfBlocks, 
                                    size_t pNumberOfThreads, size_t pShingleFactor, 
                                    size_t pBlockSizeShingle,
                                    vvsize_t_p* pSignatures);
    void copyDataToGpu(const SparseMatrixFloat* pRawData);
    void computeHitsOnGpu(std::vector<vvsize_t_p*>* pHitsPerInstance, 
                                                neighborhood* pNeighborhood, 
                                                size_t pNeighborhoodSize,
                                                size_t pNumberOfInstances,
                                                const size_t pNumberOfBlocksHistogram,
                                                const size_t pNumberOfThreadsHistogram,
                                                const size_t pNumberOfBlocksDistance,
                                                const size_t pNumberOfThreadsDistance,
                                                size_t pFast, size_t pDistance,
                                                size_t pExcessFactor, size_t pMaxNnz);
};
#endif // INVERSE_INDEX_CUDA_H 