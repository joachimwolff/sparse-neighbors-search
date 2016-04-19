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

#include "nearestNeighborsCuda.h"
#include "kernel.h"
#include "typeDefinitionsBasic.h"


NearestNeighborsCuda::NearestNeighborsCuda() {
    
}

NearestNeighborsCuda::~NearestNeighborsCuda() {
    
}

neighborhood* NearestNeighborsCuda::computeNearestNeighbors(neighborhood* neighbors, size_t pSimilarity, SparseMatrixFloat* pRawData) {
    // if pRawData == null set pointers to original data
    // else load new data to gpu
    //      compute dotProducts
    float** precomputedDotProductInstance;
    int** featureIdsInstance;
    float** valuesInstance;
    size_t maxNnzInstance;
    size_t** sizeInstance;
    if (pRawData == NULL) {
        precomputedDotProductInstance = mDev_DotProducts;
        featureIdsInstance = mDev_FeatureList;
        valuesInstance = mDev_ValuesList;
        sizeInstance = mDev_SizeOfInstanceList;
        maxNnzInstance = mMaxNnz;
        
    } else {
        maxNnzInstance = pRawData->getMaxNnz();
        cudaMalloc((void **) &(*precomputedDotProductInstance), sizeof(float) * pRawData->size());
        cudaMalloc((void **) &(*featureIdsInstance), sizeof(int) * pRawData->size() * pRawData->getMaxNnz());
        cudaMalloc((void **) &(*valuesInstance), sizeof(float) * pRawData->size() * pRawData->getMaxNnz());
        cudaMalloc((void **) &(*sizeInstance), sizeof(size_t) * pRawData->size());
        
        cudaMemcpy((*featureIdsInstance), pRawData->getSparseMatrixIndex(),
                pRawData->size() * pRawData->getMaxNnz() * sizeof(int),
            cudaMemcpyHostToDevice);
    
        cudaMemcpy((*valuesInstance), pRawData->getSparseMatrixValues(),
                pRawData->size() * pRawData->getMaxNnz() * sizeof(float),
            cudaMemcpyHostToDevice);
        cudaMemcpy((*sizeInstance), pRawData->getSparseMatrixSizeOfInstances(),
            sizeof(size_t) * pRawData->size(),
            cudaMemcpyHostToDevice);   
        dotProductSingle<<<128, 128>>>(*featureIdsInstance, *valuesInstance, *sizeInstance, 
                                        pRawData->size(), pRawData->getMaxNnz(), *precomputedDotProductInstance);
    }
    // compute dotproducts for all pairs
    
    size_t* jumpLengthList = (size_t*) malloc(neighbors->neighbors->size() * sizeof(size_t));
    size_t count = 0;
    size_t jumpLength = 0;
    size_t* candidatesSize = (size_t*) malloc(neighbors->neighbors->size() * sizeof(size_t));
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        jumpLengthList[i] = jumpLength;
        count += neighbors->neighbors->operator[](i).size();
        jumpLength += count;
        candidatesSize[i] = neighbors->neighbors->operator[](i).size();
    }
    float3* dotProducts;
    cudaMalloc((void **) &dotProducts, sizeof(float3) * count);
    int* candidates = (int*) malloc(count * sizeof(int));
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
            candidates[jumpLengthList[i]+j] = neighbors->neighbors->operator[](i)[j];
        }
    } 
    int* candidatesCuda;
    cudaMalloc((void **) &candidatesCuda, count * sizeof(int));
    cudaMemcpy(candidatesCuda, candidates, count * sizeof(int), cudaMemcpyHostToDevice);
    size_t* jumpLengthListCuda;
    cudaMalloc((void **) &jumpLengthListCuda, neighbors->neighbors->size() * sizeof(size_t));
    cudaMemcpy(jumpLengthListCuda, jumpLengthList, neighbors->neighbors->size() * sizeof(size_t), cudaMemcpyHostToDevice);
    size_t* candidatesSizeCuda;
    cudaMalloc((void **) &candidatesSizeCuda, neighbors->neighbors->size() * sizeof(size_t));
    cudaMemcpy(candidatesSizeCuda, candidatesSize, neighbors->neighbors->size() * sizeof(size_t), cudaMemcpyHostToDevice);
    // call computDotProducts
    computeDotProducts<<<128, 128>>>(dotProducts, count, candidatesCuda, jumpLengthListCuda,
                                      candidatesSizeCuda, *mDev_FeatureList, *mDev_ValuesList,
                                      mMaxNnz, *mDev_SizeOfInstanceList,
                                      *featureIdsInstance, *valuesInstance, maxNnzInstance,
                                      *sizeInstance, *mDev_DotProducts, *precomputedDotProductInstance);
    float* resultsCuda;
    cudaMalloc((void **) &resultsCuda, sizeof(float) * count);
    // compute euclidean distance or cosine similarity
    if (pSimilarity) {
        
    } else {
        euclideanDistanceCuda<<<128, 128>>>(dotProducts, count, resultsCuda);
    }
     // copy data back and sort
    float* results = (float*) malloc( sizeof(float) * count);
    cudaMemcpy(results, resultsCuda, sizeof(float) * count, cudaMemcpyDeviceToHost);
    neighborhood* neighbors_;
    neighbors_->neighbors = new vvsize_t(neighbors->neighbors->size());
    neighbors_->distances = new vvfloat(neighbors->neighbors->size());
    
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        std::vector<sortMapFloat> returnValue(neighbors->neighbors->operator[](i).size());
        
        for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
            sortMapFloat element; 
            element.key = neighbors->neighbors->operator[](i)[j];
            element.val = results[jumpLengthList[i]+j];
            returnValue[j] = element;
        }
        if (pSimilarity) {
            std::sort(returnValue.begin(), returnValue.end(), mapSortDescByValueFloat);
        } else {
            std::sort(returnValue.begin(), returnValue.end(), mapSortAscByValueFloat);
        }
        for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
            neighbors_->neighbors->operator[](i)[j] = returnValue[j].key;
            neighbors_->distances->operator[](i)[j] = returnValue[j].val;
        }
    } 
    
    // free memory
    
   
    return neighbors_;
}