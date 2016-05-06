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
//    int** mDev_FeatureList;
//    float** mDev_ValuesList;
//    size_t** mDev_SizeOfInstanceList;
//    size_t mMaxNnz;
//    size_t** mDev_JumpLengthList;
//    float** mDev_DotProducts;
    
  public:
    NearestNeighborsCuda();
  	// NearestNeighborsCuda(int* pFeatureList, float* pValuesList,
    //                       int* pSizeOfInstanceList);
    ~NearestNeighborsCuda();
    neighborhood* computeNearestNeighbors(neighborhood* neighbors, 
                size_t pSimilarity, SparseMatrixFloat* pRawData, 
                SparseMatrixFloat* pOriginalRawData, size_t pMaxNeighbors);
    // void setFeatureList(int** pFeatureList) {
    //     mDev_FeatureList = pFeatureList;
    // };
    // void setValuesList(float** pValuesList) {
    //     mDev_ValuesList = pValuesList;
    // };
    // void setSizeOfInstanceList(size_t** pSizeOfInstanceList) {
    //     mDev_SizeOfInstanceList = pSizeOfInstanceList;
        
    // };
    // void setMaxNnz(size_t pMaxNnz) {
    //     mMaxNnz = pMaxNnz;
    // };
    // void setJumpLenthList(size_t** pJumpLengthList) {
    //     mDev_JumpLengthList = pJumpLengthList;
    // }
    // void setDotProduct(float** pDotProduct) {
    //     mDev_DotProducts = pDotProduct;
    // };
};
#endif // NEAREST_NEIGHBORS_CUDA_H 