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

neighborhood* NearestNeighborsCuda::computeNearestNeighbors(neighborhood* neighbors, size_t pSimilarity, SparseMatrixFloat* pRawData,
                                                            SparseMatrixFloat* pOriginalRawData, size_t pMaxNeighbors) {
    float* precomputedDotProductNeighbor;
    int* featureIdsNeighbor;
    float* valuesNeighbor;
    size_t maxNnzNeighbor;
    size_t* sizeNeighbor;
    
    float* precomputedDotProductInstance;
    int* featureIdsInstance;
    float* valuesInstance;
    size_t maxNnzInstance;
    size_t* sizeInstance;
   
    // transfer data to gpu and precompute the dot product
    maxNnzNeighbor = pOriginalRawData->getMaxNnz();
    cudaMalloc((void **) &precomputedDotProductNeighbor, sizeof(float) * pOriginalRawData->size());
    cudaMalloc((void **) &featureIdsNeighbor, sizeof(int) * pOriginalRawData->size() * pOriginalRawData->getMaxNnz());
    cudaMalloc((void **) &valuesNeighbor, sizeof(float) * pOriginalRawData->size() * pOriginalRawData->getMaxNnz());
    cudaMalloc((void **) &sizeNeighbor, sizeof(size_t) * pOriginalRawData->size());
    
    cudaMemcpy(featureIdsNeighbor, pOriginalRawData->getSparseMatrixIndex(),
            pOriginalRawData->size() * pOriginalRawData->getMaxNnz() * sizeof(int),
        cudaMemcpyHostToDevice);

    cudaMemcpy(valuesNeighbor, pOriginalRawData->getSparseMatrixValues(),
            pOriginalRawData->size() * pOriginalRawData->getMaxNnz() * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaMemcpy(sizeNeighbor, pOriginalRawData->getSparseMatrixSizeOfInstances(),
        sizeof(size_t) * pOriginalRawData->size(),
        cudaMemcpyHostToDevice);  
    cudaDeviceSynchronize();

    dotProductSingle<<<1024, 32>>>(featureIdsNeighbor, valuesNeighbor, sizeNeighbor, 
                                    pOriginalRawData->size(), pOriginalRawData->getMaxNnz(), precomputedDotProductNeighbor);
    cudaDeviceSynchronize();
    
    if (pRawData == NULL) {
        precomputedDotProductInstance = precomputedDotProductNeighbor;
        featureIdsInstance = featureIdsNeighbor;
        valuesInstance = valuesNeighbor;
        maxNnzInstance = maxNnzNeighbor;
        sizeInstance = sizeNeighbor;
    } else {
        // the query dataset is different from the fitted one
        // transfer data to gpu and precompute the dot product
        
        maxNnzInstance = pRawData->getMaxNnz();
        cudaMalloc((void **) &precomputedDotProductInstance, sizeof(float) * pRawData->size());
        cudaMalloc((void **) &featureIdsInstance, sizeof(int) * pRawData->size() * pRawData->getMaxNnz());
        cudaMalloc((void **) &valuesInstance, sizeof(float) * pRawData->size() * pRawData->getMaxNnz());
        cudaMalloc((void **) &sizeInstance, sizeof(size_t) * pRawData->size());
        
        cudaMemcpy(featureIdsInstance, pRawData->getSparseMatrixIndex(),
                pRawData->size() * pRawData->getMaxNnz() * sizeof(int),
            cudaMemcpyHostToDevice);
    
        cudaMemcpy(valuesInstance, pRawData->getSparseMatrixValues(),
                pRawData->size() * pRawData->getMaxNnz() * sizeof(float),
            cudaMemcpyHostToDevice);
        cudaMemcpy(sizeInstance, pRawData->getSparseMatrixSizeOfInstances(),
            sizeof(size_t) * pRawData->size(),
            cudaMemcpyHostToDevice);   
        cudaDeviceSynchronize();

        dotProductSingle<<<1024, 32>>>(featureIdsInstance, valuesInstance, sizeInstance, 
                                        pRawData->size(), pRawData->getMaxNnz(), precomputedDotProductInstance);
        cudaDeviceSynchronize();
    }
    // compute jump lenghts for list of neighbor candidates.
    // transfer data to gpu and create space for the euclidean distance/cosine similarity computation
    // --> float3* dotProducts
    size_t* jumpLengthList = (size_t*) malloc(neighbors->neighbors->size() * sizeof(size_t));
    size_t count = 0;
    size_t* candidatesSize = (size_t*) malloc(neighbors->neighbors->size() * sizeof(size_t));
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        jumpLengthList[i] = count;
        count += neighbors->neighbors->operator[](i).size();
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
    // compute all dot products for all candidates with their specific query instance.
    // The base dataset is called 'neighbors' the query instances are 'instance'
    computeDotProducts<<<1024, 32>>>(dotProducts, count, candidatesCuda, jumpLengthListCuda,
                                      candidatesSizeCuda, neighbors->neighbors->size(), featureIdsNeighbor, valuesNeighbor,
                                      maxNnzNeighbor, sizeNeighbor,
                                      featureIdsInstance, valuesInstance, maxNnzInstance,
                                      sizeInstance, precomputedDotProductNeighbor, precomputedDotProductInstance);
    cudaDeviceSynchronize();
    
    float* resultsCuda;
    cudaMalloc((void **) &resultsCuda, sizeof(float) * count);
    // compute euclidean distance or cosine similarity
    if (pSimilarity) { 
        cosineSimilarityCuda<<<1024, 32>>>(dotProducts, count, resultsCuda);
    } else {
        euclideanDistanceCuda<<<1024, 32>>>(dotProducts, count, resultsCuda);
    }
     // copy data back and sort
    float* results = (float*) malloc( sizeof(float) * count);
    cudaMemcpy(results, resultsCuda, sizeof(float) * count, cudaMemcpyDeviceToHost);
    
    // return results
    neighborhood* neighbors_ = new neighborhood();;
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
        size_t vectorSize = std::min(returnValue.size(), pMaxNeighbors);
        if (pSimilarity) {
            std::partial_sort(returnValue.begin(), returnValue.begin()+vectorSize, returnValue.end(), mapSortDescByValueFloat);
        } else {
            std::partial_sort(returnValue.begin(), returnValue.begin()+vectorSize, returnValue.end(), mapSortAscByValueFloat);
        }
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
    
    cudaFree(dotProducts);
    cudaFree(candidatesCuda);
    cudaFree(jumpLengthListCuda);
    cudaFree(candidatesSizeCuda);
    cudaFree(featureIdsNeighbor);
    cudaFree(valuesNeighbor);
    cudaFree(sizeNeighbor);
    cudaFree(precomputedDotProductNeighbor);
    if (pRawData != NULL) {
        cudaFree(featureIdsInstance);
        cudaFree(valuesInstance);
        cudaFree(sizeInstance);
        cudaFree(precomputedDotProductInstance);
    }
    free(jumpLengthList);
    free(candidates);
    free(candidatesSize);
    free(results);
    cudaDeviceSynchronize();
    
    return neighbors_;
}