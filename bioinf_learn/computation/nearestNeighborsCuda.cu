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
                                                            SparseMatrixFloat* pOriginalRawData) {
    // if pRawData == null set pointers to original data
    // else load new data to gpu
    //      compute dotProducts
        // printf("%i\n", __LINE__);


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
   
    // printf("%i\n", __LINE__);
    
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
    // computeDotProducts<<<128, 128>>>(sizeNeighbor);
             cudaDeviceSynchronize();

    dotProductSingle<<<1024, 32>>>(featureIdsNeighbor, valuesNeighbor, sizeNeighbor, 
                                    pOriginalRawData->size(), pOriginalRawData->getMaxNnz(), precomputedDotProductNeighbor);
       cudaDeviceSynchronize();
    // float* dotCuda = (float*) malloc( pOriginalRawData->size() * sizeof(float));
    // cudaMemcpy(dotCuda, precomputedDotProductNeighbor, sizeof(float) * pOriginalRawData->size(), cudaMemcpyDeviceToHost);
    // printf("Precomputed dotptoducts: \n");
    // for (size_t  i = 0; i <  pOriginalRawData->size(); ++i) {
    //     printf("%f\n", dotCuda[i]);
    // }
    // printf("\n\n"); 
    
    if (pRawData == NULL) {
        precomputedDotProductInstance = precomputedDotProductNeighbor;
        featureIdsInstance = featureIdsNeighbor;
        valuesInstance = valuesNeighbor;
        maxNnzInstance = maxNnzNeighbor;
        sizeInstance = sizeNeighbor;
    
    // float* precomputedDotProductNeighbor;
    // int* featureIdsNeighbor;
    // float* valuesNeighbor;
    // size_t maxNnzNeighbor;
    // size_t* sizeNeighbor;
    } else {
       
   
        // printf("%i\n", __LINE__);
        
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
    // compute dotproducts for all pairs
        // printf("%i\n", __LINE__);
    
    size_t* jumpLengthList = (size_t*) malloc(neighbors->neighbors->size() * sizeof(size_t));
    size_t count = 0;
    // size_t jumpLength = 0;
    size_t* candidatesSize = (size_t*) malloc(neighbors->neighbors->size() * sizeof(size_t));
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        jumpLengthList[i] = count;
        count += neighbors->neighbors->operator[](i).size();
        // printf("count: %i\n", count);
       
        candidatesSize[i] = neighbors->neighbors->operator[](i).size();
        // printf("%u: %u\n", i, candidatesSize[i]);
    }
        // printf("%i\n", __LINE__);
    
    float3* dotProducts;
        // printf("%i\n", __LINE__);
    
    cudaMalloc((void **) &dotProducts, sizeof(float3) * count);
        // printf("count: %i, %i\n", count, __LINE__);
    
    int* candidates = (int*) malloc(count * sizeof(int));
        // printf("%i\n", __LINE__);
        // printf("count: %i\n", count);
        
    
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
            // printf("jumpLengthList[i]: %i + j: %i\n", jumpLengthList[i], j);
            candidates[jumpLengthList[i]+j] = neighbors->neighbors->operator[](i)[j];
            // printf("%i, ", candidates[jumpLengthList[i]+j]);

        }
        // printf("%i: %i\n", i, jumpLengthList[i]);
    } 
        // printf("count: %i, %i\n", count, __LINE__);
    
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
    // printf("%i\n", __LINE__); 
    // printf("pMaxNnzNeighbor: %u, pMaxNnzInstance: %u\n", maxNnzNeighbor, maxNnzInstance);
    computeDotProducts<<<1024, 32>>>(dotProducts, count, candidatesCuda, jumpLengthListCuda,
                                      candidatesSizeCuda, neighbors->neighbors->size(), featureIdsNeighbor, valuesNeighbor,
                                      maxNnzNeighbor, sizeNeighbor,
                                      featureIdsInstance, valuesInstance, maxNnzInstance,
                                      sizeInstance, precomputedDotProductNeighbor, precomputedDotProductInstance);
    
    // computeDotProducts<<<128, 128>>>(sizeNeighbor);
    //    printf("DotproductNeighbors: %u\n", mDev_DotProducts);
    cudaDeviceSynchronize();

    // float3* dotProductsResult = (float3*) malloc(sizeof(float3) *count);
    // cudaMemcpy(dotProductsResult, dotProducts, sizeof(float) * count, cudaMemcpyDeviceToHost);
    // for (size_t i = 0; i < count; ++i) {
    //     printf("%f, %f, %f\n", dotProductsResult[i].x, dotProductsResult[i].y, dotProductsResult[i].z);
    // }
  
    float* resultsCuda;
    cudaMalloc((void **) &resultsCuda, sizeof(float) * count);
    // compute euclidean distance or cosine similarity
    if (pSimilarity) { 
        // printf("%i\n", __LINE__);
        
    } else {
        // printf("%i\n", __LINE__);
        
        euclideanDistanceCuda<<<1024, 32>>>(dotProducts, count, resultsCuda);
    }
        // printf("%i\n", __LINE__);
    
     // copy data back and sort
    float* results = (float*) malloc( sizeof(float) * count);
        // printf("%i\n", __LINE__);
    
    cudaMemcpy(results, resultsCuda, sizeof(float) * count, cudaMemcpyDeviceToHost);
        // printf("%i\n", __LINE__);
    // return results;
    neighborhood* neighbors_ = new neighborhood();;
    neighbors_->neighbors = new vvsize_t(neighbors->neighbors->size());
    neighbors_->distances = new vvfloat(neighbors->neighbors->size());
        // printf("%i\n", __LINE__);
    // #pragma omp parallel for
    for (size_t i = 0; i < neighbors->neighbors->size(); ++i) {
        std::vector<sortMapFloat> returnValue(neighbors->neighbors->operator[](i).size());
            // if (i == 4336)
            //   printf("%i\n", __LINE__);
        
        for (size_t j = 0; j < neighbors->neighbors->operator[](i).size(); ++j) {
            sortMapFloat element; 
            element.key = neighbors->neighbors->operator[](i)[j];
            element.val = results[jumpLengthList[i]+j];
            // if (i == 0)
            // printf("%u:%f, ",element.key, element.val);
            returnValue[j] = element;
        }
        // if (i == 0)
        // printf("\n\n");
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
                // if (i == 0)
                //     printf("%u:%f\n", returnValue[j].key, returnValue[j].val);
        }
        neighbors_->neighbors->operator[](i) = neighborsVector;
        neighbors_->distances->operator[](i) = distancesVector;
    } 
    
    // free memory
    
        // printf("%i\n", __LINE__);
   
//    dotProducts, count, candidatesCuda, jumpLengthListCuda,
//                                       candidatesSizeCuda, featureIdsNeighbor, valuesNeighbor,
//                                       maxNnzNeighbor, sizeNeighbor,
//                                       featureIdsInstance, valuesInstance, maxNnzInstance,
//                                       sizeInstance, precomputedDotProductNeighbor, precomputedDotProductInstance
   cudaFree(dotProducts);
   cudaFree(candidatesCuda);
   cudaFree(jumpLengthListCuda);
   cudaFree(candidatesSizeCuda);
   cudaFree(featureIdsNeighbor);
   cudaFree(valuesNeighbor);
   cudaFree(sizeNeighbor);
   cudaFree(featureIdsInstance);
   cudaFree(valuesInstance);
   cudaFree(sizeInstance);
   cudaFree(precomputedDotProductNeighbor);
   cudaFree(precomputedDotProductInstance);
//    cudaFree();
   
   
    return neighbors_;
}