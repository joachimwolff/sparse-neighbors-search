#include <stdio.h>
#include "sparseMatrix.h"
#include <math.h>
#define float A = sqrt(2) - 1;

__global__ void fit(size_t* pFeatureIdList, size_t* pSizeOfInstanceList, size_t pNumberOfHashFunctions, size_t pMaxNnz
                        size_t* pComputedSignatures) {
    __shared__ size_t signature[pNumberOfHashFunctions];
    size_t threadId = threadIdx.x;
    size_t blockId = blockIdx.x;
    size_t minHashValue = MAX_VALUE;
    size_t hashValue = 0;
    if (blockId < pRawData->size()) {
        // size_t sizeOfInstance = pSizeOfInstanceList[blockId];
        for (size_t i = 0; i < pSizeOfInstanceList[blockId]; ++i) {
            hashValue = computeHashValue((pFeatureIdList[blockId*pMaxNnz + i] + 1)*threadId, MAX_VALUE);
            if (hashValue < minHashValue) {
                minHashValue = hashValue;
            }
        }
        signature[threadId] = minHashValue;
    }
    __syncthread();
    
    if (threadId == 0) {
        for (size_t i = 0; i < pNumberOfHashFunctions; ++i) {
            pComputedSignatures[blockId*pNumberOfHashFunctions +i] = signature[i];
        }
    }
}

__device__ size_t computeHashValue(size_t key, size_t aModulo) {
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

__global__ void query(size_t** pSignature, size_t** pInverseIndex, size_t* pElementsPerHashFunction, size_t pNumberOfHashFunctions,
                        size_t* pHitsToBeReturned) {
    size_t threadId = threadIdx.x;
    size_t blockId = blockIdx.x;
    size_t hashValue = pSignature[blockId][threadId];
    __shared__ size_t hits[pNumberOfHashFunctions];
    
    for (size_t i = 0; i < pElementsPerHashFunction[threadId]; ++i) {
        if (hashValue == pInverseIndex[threadId][i]) {
            hits[threadId] = i;
        } else if (hashValue > pInverseIndex[threadId][i]) {
            hits[threadId] = MAX_VALUE;
            continue;
        }
    }
    __syncthread();
    
    if (threadId == 0) {
        for (size_t i = 0; i < pNumberOfHashFunctions; ++i) {
            pHitsToBeReturned[blockId*pNumberOfHashFunctions + i] = hits[i];
        }
    }  
}

__global__ void euclidianDistance(size_t* pFeatureIds, size_t* pSizeOfInstanceList,
                                    float* pFeatureValues, size_t pMaxNnz,
                                    size_t* pPossibleInstances, size_t* pSizePerInstance, size_t pMaxCandidates,                             
                                    size_t* pHitsToBeReturned, float* pValuesToBeReturned) {
    size_t threadId = threadIdx.x;
    size_t blockId = blockIdx.x;
    // size_t hashValue = pSignature[blockId][threadId];
    // __shared__ size_t hits[pNumberOfHashFunctions];
    size_t startIndexInstance = *  
    size_t endIndexInstance = 
    
}

__global__ void cosineSimilarity() {
    
}