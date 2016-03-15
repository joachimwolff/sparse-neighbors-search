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
   size_t* mDev_FeatureList;
   float* mDev_ValuesList;
   size_t* mDev_SizeOfInstanceList;
   size_t mMaxNnz;
    
  public:
    NearestNeighborsCuda();
  	NearestNeighborsCuda(size_t* pFeatureList, float* pValuesList,
                          size_t* pSizeOfInstanceList, size_t pMaxNnz);
    ~NearestNeighborsCuda();
    cudaInstanceVector* computeNearestNeighbors(neighborhood* neighbors, size_t pSimilarity);
    void setFeatureList(size_t* pFeatureList) {
        mDev_FeatureList = pFeatureList;
    };
    void setValuesList(float* pValuesList) {
        mDev_ValuesList = pValuesList;
    };
    void setSizeOfInstanceList(size_t* pSizeOfInstanceList) {
        mDev_SizeOfInstanceList = pSizeOfInstanceList;
    };
    void setMaxNnz(size_t pMaxNnz) {
        mMaxNnz = pMaxNnz;
    };
};
#endif // NEAREST_NEIGHBORS_CUDA_H 