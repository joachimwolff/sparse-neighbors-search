#include <stdio.h>
#include "sparseMatrix.h"
// #include <math.h>
#include "kernel.h"

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
                    const size_t pNumberOfInstances) {
    extern __shared__ size_t signature[];  // pNumberOfHashFunctions
    size_t threadId = threadIdx.x;
    size_t blockId = blockIdx.x;
    size_t minHashValue = MAX_VALUE;
    size_t hashValue = 0;
    if (blockId < pNumberOfInstances*pMaxNnz) {
        // size_t sizeOfInstance = pSizeOfInstanceList[blockId];
        for (size_t i = 0; i < pSizeOfInstanceList[blockId]; ++i) {
            hashValue = computeHashValueCuda((pFeatureIdList[blockId*pMaxNnz + i] + 1)*threadId, MAX_VALUE);
            if (hashValue < minHashValue) {
                minHashValue = hashValue;
            }
        }
        signature[threadId] = minHashValue;
    }
    __syncthreads();
    
    if (threadId == 0) {
        for (size_t i = 0; i < pNumberOfHashFunctions; ++i) {
            pComputedSignatures[blockId*pNumberOfHashFunctions +i] = signature[i];
        }
    }
}



__global__ void queryCuda(size_t* pSignature, size_t* pInverseIndex, size_t pNumberOfHashFunctions, 
                        size_t pNumberOfInstancs, size_t pSizeOfInverseIndex, size_t pSignatureCount,
                        size_t* pHitsToBeReturned) {
    extern __shared__ size_t hits[]; // pNumberOfHashFunctions
    size_t inverseIndexId;
    size_t signatureId;
    size_t hit;
    size_t signatureCount = 1;
    // do query for every signature
    while (signatureCount <= pSignatureCount) {
        inverseIndexId = blockIdx.x*blockDim.x;
        signatureId = threadIdx.x * signatureCount;
        
        // compare every value of hash function i of the signature with the 
        // value of hash function i of every instance.
        // In the optimale case this is done in O(1) based on parallelism
        // if per grid only one block and one thread is launched, 
        // it is computed in O(n^2)
        while (inverseIndexId < pSizeOfInverseIndex) {
            while (signatureId < pNumberOfHashFunctions) {
                hit = 0;
                if (pInverseIndex[inverseIndexId + signatureId] == pSignature[signatureId]) {
                    hit = 1;
                } 
                hits[signatureId / signatureCount] = hit;
                signatureId += blockDim.x;
            }
            inverseIndexId += blockDim.x * gridDim.x;
        }
        
        __syncthreads();
        
        // reduction
        signatureId = threadIdx.x;
        int i = blockDim.x / 2;
        int tmp_i;
        while (i != 0) {
            tmp_i = i;
            while (signatureId < pNumberOfHashFunctions) {
                if (signatureId < i) {
                    hits[signatureId] += hits[signatureId + i];
                }
                signatureId += blockDim.x;
                i += blockDim.x;
            }
            __syncthreads();
            i = tmp_i;
            i /= 2;
            signatureId = threadIdx.x;
        }
        __syncthreads();
        
        if (signatureId == 0) {
            pHitsToBeReturned[blockIdx.x * signatureCount] = hits[0];
        }  
        ++signatureCount;
    }
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