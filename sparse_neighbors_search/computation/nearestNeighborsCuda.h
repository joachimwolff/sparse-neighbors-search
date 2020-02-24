/**
 Copyright 2016, 2017, 2018, 2019, 2020 Joachim Wolff
 PhD Thesis

 Copyright 2015, 2016 Joachim Wolff
 Master Thesis
 Tutor: Fabrizio Costa
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
    
  public:
    NearestNeighborsCuda();
    ~NearestNeighborsCuda();
    neighborhood* computeNearestNeighbors(neighborhood* neighbors, 
                size_t pSimilarity, SparseMatrixFloat* pRawData, 
                SparseMatrixFloat* pOriginalRawData, size_t pMaxNeighbors);
};
#endif // NEAREST_NEIGHBORS_CUDA_H 