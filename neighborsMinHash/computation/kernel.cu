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

#include <stdio.h>
#include "sparseMatrix.h"
// #include <math.h>
#include "kernel.h"
#include <cub/cub.cuh>
__device__ size_t computeHashValueCuda(size_t key, size_t aModulo) {
    // source:  Thomas Wang: Integer Hash Functions, 1997 / 2007 
    // https://gist.github.com/badboy/6267743
    key = key * A;
    key = ~key + (key << 15);
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;
    key = key ^ (key >> 16);
    return key % aModulo;
}

__global__ void fitCuda(const size_t* pFeatureIdList, const size_t* pSizeOfInstanceList,
                    const size_t pNumberOfHashFunctions, const size_t pMaxNnz,
                    size_t* pComputedSignatures, 
                    const size_t pNumberOfInstances, const size_t pStartInstance, 
                    const size_t pBlockSize, const size_t pShingleSize) {
    // extern __shared__ size_t signature[];  // pNumberOfHashFunctions
    int instanceId = blockIdx.x + pStartInstance;
    size_t minHashValue = MAX_VALUE;
    size_t hashValue = 0;
    
    int featureId = blockIdx.x * pMaxNnz;
    int hashFunctionId = threadIdx.x;
    size_t sizeOfInstance;
    
    while (instanceId < pNumberOfInstances) {
        
        sizeOfInstance = pSizeOfInstanceList[instanceId];
        while (hashFunctionId < pNumberOfHashFunctions && featureId < pNumberOfInstances*pMaxNnz) {
            for (size_t i = 0; i < sizeOfInstance; ++i) {
                hashValue = computeHashValueCuda((pFeatureIdList[featureId + i]+1) * (hashFunctionId+1), MAX_VALUE);
                if (hashValue < minHashValue) {
                    minHashValue = hashValue;
                }
            }
            pComputedSignatures[(instanceId-pStartInstance)*pNumberOfHashFunctions + hashFunctionId] = minHashValue;
            hashFunctionId += blockDim.x;
        }
        instanceId += gridDim.x;
        featureId = instanceId * pMaxNnz;
        minHashValue = MAX_VALUE;
        hashFunctionId = threadIdx.x;
    }
}



__global__ void queryCuda(size_t* pHitsPerInstance, size_t* pSizePerInstance,
                            size_t pNeighborhoodSize, size_t* pNeighborhood,
                            float* pDistances, const size_t pNumberOfInstances,
                            size_t* pHistogramMemory) {
    // sort hits per instances
    // count instances
    // take highest pNeighborhood*excessfaktor + same hits count
    // to compute euclidean distance or cosine similarity
    
    
    // per block query one instance
    // sort these with the threads
    
    
    const int numberOfThreads = blockDim.x;
    int instanceId = blockIdx.x;
    int threadId = threadIdx.x;
    int startId;
    int endId;
    // create histogram
    while (instanceId < pNumberOfInstances) {
        // compute start position in array pHitsPerInstance
        startId = instanceId;
        for (size_t i = 0; i < instanceId; ++i) {
            startId += pSizePerInstance[i];
        }
        endId = startId+pSizePerInstance[instanceId];
        
        while (startId + threadId < endId) {
            atomicAdd(pHistogramMemory[pHitsPerInstance[startId+threadId] * instanceId], 1);
            instanceId += gridDim.x;
            threadId += numberOfThreads;
        }
        instanceId += gridDim.x;
        threadId = threadIdx.x;
    }
    
    __syncthreads();
    int mergeInstances = 2;
    int idPart1 = 0;
    int idPart2 = 0;
    
    // radix sort histogram
    while () {
        for (size_t i = 0; i < mergeInstances; ++i) {
            if (histogram[threadId+idPart1] < histogram[threadId+idPart2]) {
                
            }
    }
    __syncthreads();
    
    // take largest values
    // compute euclidean distance or cosine similarity
    
}

__global__ void euclidianDistanceCuda(size_t* pFeatureIds, size_t* pSizeOfInstanceList,
                                    float* pFeatureValues, size_t pMaxNnz,
                                    size_t* pPossibleInstances, size_t* pSizePerInstance, size_t pMaxCandidates,                             
                                    size_t* pHitsToBeReturned, float* pValuesToBeReturned) {
    // extern __shared__ float euclidianDistance [];
    // size_t queryId;
    // size_t instanceId;
    // float queryValue;
    // float instanceValue;
    // float value;
    // size_t accessIdQuery;
    // size_t accessIdInstance;
    // size_t queryCount = 0;
    
    // // while (queryCount < )
    
    // queryId = pFeatureIds[accessIdQuery];
    // instanceId = pFeatureIds[accessIdInstance];
    // queryValue = pFeatureValues[accessIdQuery];
    // instanceValue = pFeatureValues[accessIdInstance];
    
    // if (queryId == instanceId) {
    //     value = queryValue - instanceValue;
    // } else if (queryId < instanceId) {
    //     value = -instanceValue;
    // } else {
    //     value = -queryValue;
    // }
    // euclidianDistance = powf(value, 2);
    
}

__global__ void cosineSimilarityCuda() {
    
}