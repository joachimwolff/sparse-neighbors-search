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
#include "typeDefinitionsCuda.h"

#ifndef NEAREST_NEIGHBORS_CUDA_H
#define NEAREST_NEIGHBORS_CUDA_H
class NearestNeighborsCuda {

  private: 
   int** mDev_FeatureList;
   float** mDev_ValuesList;
   int** mDev_SizeOfInstanceList;
   int mMaxNnz;
   int** mDev_JumpLengthList;
   float** mDev_DotProducts;
    
  public:
    NearestNeighborsCuda();
  	// NearestNeighborsCuda(int* pFeatureList, float* pValuesList,
    //                       int* pSizeOfInstanceList);
    ~NearestNeighborsCuda();
    cudaInstanceVector* computeNearestNeighbors(neighborhood* neighbors, size_t pSimilarity, SparseMatrixFloat* pRawData);
    void setFeatureList(int** pFeatureList) {
        mDev_FeatureList = pFeatureList;
    };
    void setValuesList(float** pValuesList) {
        mDev_ValuesList = pValuesList;
    };
    void setSizeOfInstanceList(int** pSizeOfInstanceList) {
        mDev_SizeOfInstanceList = pSizeOfInstanceList;
        
    };
    void setMaxNnz(int pMaxNnz) {
        mMaxNnz = pMaxNnz;
    };
    void setJumpLenthList(int** pJumpLengthList) {
        mDev_JumpLengthList = pJumpLengthList;
    }
    void setDotProduct(float** pDotProduct) {
        mDev_DotProducts = pDotProduct;
    };
};
#endif // NEAREST_NEIGHBORS_CUDA_H 