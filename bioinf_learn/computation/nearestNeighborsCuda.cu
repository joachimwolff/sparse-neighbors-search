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
        printf("%i\n", __LINE__);

    float** precomputedDotProductInstance;
    int** featureIdsInstance;
    float** valuesInstance;
    size_t maxNnzInstance;
    size_t** sizeInstance;
    if (pRawData == NULL) {
        printf("%i\n", __LINE__);
        
        precomputedDotProductInstance = mDev_DotProducts;
        featureIdsInstance = mDev_FeatureList;
        valuesInstance = mDev_ValuesList;
        sizeInstance = mDev_SizeOfInstanceList;
        maxNnzInstance = mMaxNnz;
        
    } else {
        printf("%i\n", __LINE__);
        
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
        printf("%i\n", __LINE__);
    
    size_t* jumpLengthList = (size_t*) malloc(neighbors->neighbors->size() * sizeof(size_t));
    size_t count = 0;
    // size_t jumpLength = 0;
    size_t* candidatesSize = (size_t*) malloc(neighbors->neighbors->size() * sizeof(size_t));
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        jumpLengthList[i] = count;
        count += neighbors->neighbors->operator[](i).size();
        // printf("count: %i\n", count);
       
        candidatesSize[i] = neighbors->neighbors->operator[](i).size();
    }
        printf("%i\n", __LINE__);
    
    float3* dotProducts;
        printf("%i\n", __LINE__);
    
    cudaMalloc((void **) &dotProducts, sizeof(float3) * count);
        printf("%i\n", __LINE__);
    
    int* candidates = (int*) malloc(count * sizeof(int));
        printf("%i\n", __LINE__);
        // printf("count: %i\n", count);
        
    
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
            // printf("jumpLengthList[i]: %i + j: %i\n", jumpLengthList[i], j);
            candidates[jumpLengthList[i]+j] = neighbors->neighbors->operator[](i)[j];
            // printf("%i, ", candidates[jumpLengthList[i]+j]);

        }
        // printf("\n");
    } 
        printf("%i\n", __LINE__);
    
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
        printf("%i\n", __LINE__); 
    
    computeDotProducts<<<128, 128>>>(dotProducts, count, candidatesCuda, jumpLengthListCuda,
                                      candidatesSizeCuda, *mDev_FeatureList, *mDev_ValuesList,
                                      mMaxNnz, *mDev_SizeOfInstanceList,
                                      *featureIdsInstance, *valuesInstance, maxNnzInstance,
                                      *sizeInstance, *mDev_DotProducts, *precomputedDotProductInstance);
    float* resultsCuda;
    cudaMalloc((void **) &resultsCuda, sizeof(float) * count);
    // compute euclidean distance or cosine similarity
    if (pSimilarity) {
        printf("%i\n", __LINE__);
        
    } else {
        printf("%i\n", __LINE__);
        
        euclideanDistanceCuda<<<128, 128>>>(dotProducts, count, resultsCuda);
    }
        printf("%i\n", __LINE__);
    
     // copy data back and sort
    float* results = (float*) malloc( sizeof(float) * count);
        printf("%i\n", __LINE__);
    
    cudaMemcpy(results, resultsCuda, sizeof(float) * count, cudaMemcpyDeviceToHost);
        printf("%i\n", __LINE__);
    // return results;
    neighborhood* neighbors_ = new neighborhood();;
    neighbors_->neighbors = new vvsize_t(neighbors->neighbors->size());
    neighbors_->distances = new vvfloat(neighbors->neighbors->size());
        printf("%i\n", __LINE__);
    
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        std::vector<sortMapFloat> returnValue(neighbors->neighbors->operator[](i).size());
        // printf("%i\n", __LINE__);
        
        for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
            sortMapFloat element; 
            element.key = neighbors->neighbors->operator[](i)[j];
            element.val = results[jumpLengthList[i]+j];
            if (i+j < 32)
            printf("%f, ", element.val);
            returnValue[j] = element;
        }
        if (i == 0)
        printf("\n");
        if (pSimilarity) {
            std::sort(returnValue.begin(), returnValue.end(), mapSortDescByValueFloat);
        } else {
            std::sort(returnValue.begin(), returnValue.end(), mapSortAscByValueFloat);
        }
        size_t vectorSize = returnValue.size();
                
        std::vector<size_t> neighborsVector(vectorSize);
        std::vector<float> distancesVector(vectorSize);
        if (vectorSize == 0) {
            neighborsVector.push_back(i);
            distancesVector.push_back(0.0);
        }
        for (size_t j = 0; j < vectorSize; ++j) {
                neighborsVector[j] = returnValue[j].key;
                distancesVector[j] = returnValue[j].val;
        }
        neighbors_->neighbors->operator[](i) = neighborsVector;
        neighbors_->distances->operator[](i) = distancesVector;
    } 
    
    // free memory
    
        printf("%i\n", __LINE__);
   
    return neighbors_;
}