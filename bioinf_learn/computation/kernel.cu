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
#include "kernel.h"

__device__ int computeHashValueCuda(int pKey, int aModulo) {
    // source:  Thomas Wang: Integer Hash Functions, 1997 / 2007 
    // https://gist.github.com/badboy/6267743
    
    pKey = pKey * A;
    pKey= ~pKey + (pKey << 15);
    pKey = pKey ^ (pKey >> 12);
    pKey = pKey + (pKey << 2);
    pKey = pKey ^ (pKey >> 4);
    pKey = pKey * 2057;
    pKey = pKey ^ (pKey >> 16);
    return pKey % aModulo;
}

__global__ void fitCudaMinHash(const int* pFeatureIdList, const size_t* pSizeOfInstanceList,
                    const size_t pNumberOfHashFunctions, const size_t pMaxNnz,
                    int* pComputedSignatures, 
                    const size_t pNumberOfInstances, const size_t pStartInstance, 
                    const size_t pBlockSize, const size_t pShingleSize,
                    int* pSignaturesBlockSize) {
    int instanceId = blockIdx.x + pStartInstance;
    int nearestNeighborsValue = MAX_VALUE;
    int hashValue = 0;
    int signatureSize = pNumberOfHashFunctions * pBlockSize / pShingleSize;
    int featureId = blockIdx.x * pMaxNnz;
    int hashFunctionId = threadIdx.x;
    int sizeOfInstance;
    int signatureBlockValue;
    int shingleId;
    int signatureBlockId = blockIdx.x * pNumberOfHashFunctions * pBlockSize;
    // compute one instance per block
    // if one instance is computed, block takes next instance
    while (instanceId < pNumberOfInstances) {
        // compute the nearestNeighborsValue for every hash function
        // if pBlockSize is greater as 1, hash functions * pBlockSize values 
        // are computed. They will be merged together by a factor of pShingleSize
        sizeOfInstance = pSizeOfInstanceList[instanceId];
        while (hashFunctionId < pNumberOfHashFunctions * pBlockSize && featureId < pNumberOfInstances*pMaxNnz) {
            for (size_t i = 0; i < sizeOfInstance; ++i) {
                hashValue = computeHashValueCuda((pFeatureIdList[featureId + i]+1) * (hashFunctionId+1), MAX_VALUE);
                if (hashValue < nearestNeighborsValue) {
                    nearestNeighborsValue = hashValue;
                }
            }
            pSignaturesBlockSize[signatureBlockId + hashFunctionId] = nearestNeighborsValue;
            hashFunctionId += blockDim.x;
            nearestNeighborsValue = MAX_VALUE;
        }
        __syncthreads();
        // merge pShingleSize values together.
        // do one merge per thread
        hashFunctionId = threadIdx.x * pShingleSize;
        shingleId = threadIdx.x;
        while (hashFunctionId < pNumberOfHashFunctions * pBlockSize ) {
            signatureBlockValue = pSignaturesBlockSize[signatureBlockId + hashFunctionId];
            for (size_t i = 1; i < pShingleSize && hashFunctionId+i < pNumberOfHashFunctions * pBlockSize; ++i) {
                signatureBlockValue = computeHashValueCuda((pSignaturesBlockSize[signatureBlockId + hashFunctionId+i]+1) * (signatureBlockValue+1), MAX_VALUE);
            }
            pComputedSignatures[(instanceId-pStartInstance)*signatureSize + shingleId] = signatureBlockValue;
            hashFunctionId += blockDim.x * pShingleSize;
            shingleId += blockDim.x;
        }
        __syncthreads();
        instanceId += gridDim.x;
        featureId = instanceId * pMaxNnz;
        nearestNeighborsValue = MAX_VALUE;
        hashFunctionId = threadIdx.x;
    }
}

__global__ void fitCudaWtaHash(const int* pFeatureIdList, const int* pSizeOfInstanceList,
                    const int pNumberOfHashFunctions, const int* pJumpLengthList,
                    int* pComputedSignatures, 
                    const int pNumberOfInstances, const int pStartInstance, 
                    const int pBlockSize, const int pShingleSize,
                    int* pSignaturesBlockSize) {
                            
}



__global__ void dotProductSingle(int* pFeatureList, float* pValuesList,
                                 size_t* pSizeOfInstanceList,
                                 size_t pSize, size_t pMaxNnz, float* pDevDotProduct) {
    int instanceId = blockIdx.x;
    int threadId = threadIdx.x;
    float __shared__ value[32];
    int __shared__ jumpLength;
    size_t __shared__ size;
    
    
    while (instanceId < pSize) {
        value[threadIdx.x] = 0;
        if (threadIdx.x == 0) {
            jumpLength = instanceId * pMaxNnz;
            size = pSizeOfInstanceList[instanceId];
        }
        __syncthreads();
        while (threadId < size) {
            value[threadIdx.x] += pValuesList[jumpLength + threadId] *  pValuesList[jumpLength + threadId];
            
            threadId += blockDim.x;
        }
        // reduce
        __syncthreads();
        int i = blockDim.x/2;
        while (i != 0) {
            if (threadIdx.x < i) { 
                value[threadIdx.x] += value[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }
            
        pDevDotProduct[instanceId] = value[0];
        instanceId += gridDim.x;
        threadId = threadIdx.x;
    }                                
}


__global__ void computeDotProducts(float3* pDotProducts, size_t pSize, 
                                        int* pCandidates, size_t* pJumpLength, 
                                        size_t* pCandidateSize, size_t pNumberOfCandidates,
                                        int* pFeatureIdsNeighbor, float* pValuesNeighbor, 
                                        size_t pMaxNnzNeighbor, size_t* pSizeNeighbor,
                                        int* pFeatureIdsInstance, float* pValuesInstance,
                                        size_t pMaxNnzInstance, size_t* pSizeInstance,
                                         float* pPreComputedDotProductsNeighbor, 
                                         float* pPreComputedDotProductsInstance) {
    
      
    int instanceCandidates = blockIdx.x;
    int round = 0;
    int i = 0;
    const int threadCount = 32;
    __shared__ int instanceCounter;
    __shared__ int neighbor;
    __shared__ int instance;
    
    __shared__ int featureIdX[threadCount];
    __shared__ int featureIdY[threadCount];
    __shared__ float value[threadCount];
    __shared__ int pStartPosX;
    __shared__ int pEndPosX;
    __shared__ int pStartPosY;
    __shared__ int pEndPosY;
            
    while (instanceCandidates < pNumberOfCandidates) {
        if (threadIdx.x == 0) {
            neighbor = pCandidates[pJumpLength[instanceCandidates]];
            instanceCounter = 0;
        }
        __syncthreads();
        while (instanceCounter < pCandidateSize[neighbor]) {
             
            if (threadIdx.x == 0) {
                instance = pCandidates[pJumpLength[instanceCandidates]+instanceCounter];
                pStartPosX = neighbor*pMaxNnzNeighbor;
                pEndPosX = neighbor*pMaxNnzNeighbor + pSizeNeighbor[neighbor];
                pStartPosY = instance*pMaxNnzInstance;
                pEndPosY = instance*pMaxNnzInstance + pSizeInstance[instance];
            }
            value[threadIdx.x] = 0.0;
            
            __syncthreads();
            
            while (pStartPosX < pEndPosX+threadCount - (pEndPosX%threadCount) && pStartPosY < pEndPosY+threadCount - (pEndPosY%threadCount) ) {
           
                featureIdX[threadIdx.x] = pFeatureIdsNeighbor[pStartPosX + threadIdx.x];
                featureIdY[threadIdx.x] = pFeatureIdsInstance[pStartPosY + threadIdx.x];
                
                while (round < threadCount) {
                    if (featureIdX[(threadIdx.x + round) % threadCount] == featureIdY[threadIdx.x]) {   
                        value[threadIdx.x] += pValuesNeighbor[pStartPosX + ((threadIdx.x + round) % threadCount)] * pValuesInstance[pStartPosY + threadIdx.x];
                        break;
                    }
                    ++round;
                }
                __syncthreads();
                round = 0;
                if (threadIdx.x == 0) {
                    if (featureIdX[threadCount-1] == featureIdY[threadCount-1]) {
                        pStartPosY += threadCount;
                        pStartPosX += threadCount;
                    } else if (featureIdX[threadCount-1] < featureIdY[threadCount-1]) {
                        pStartPosX += threadCount;
                    } else {
                        pStartPosY += threadCount;
                    }
                }
                __syncthreads();
            }
            __syncthreads();
            
            i = blockDim.x/2;
            while (i != 0) {
                if (threadIdx.x < i) { 
                    value[threadIdx.x] += value[threadIdx.x + i];
                }
                __syncthreads();
                i /= 2;
            }
            if (threadIdx.x == 0) {
                pDotProducts[pJumpLength[instanceCandidates]+instanceCounter].y = value[0];
                pDotProducts[pJumpLength[instanceCandidates]+instanceCounter].x = pPreComputedDotProductsNeighbor[neighbor];
                pDotProducts[pJumpLength[instanceCandidates]+instanceCounter].z = pPreComputedDotProductsInstance[instance];          
                ++instanceCounter;
            }
            __syncthreads();
        }
        instanceCandidates += gridDim.x;
    }
}
__global__ void euclideanDistanceCuda(float3* pDotProducts, size_t pSize, float* results) {
  int instance = blockIdx.x * blockDim.x + threadIdx.x;
  
  while (instance < pSize) {
      results[instance] = pDotProducts[instance].x - 2*pDotProducts[instance].y + pDotProducts[instance].z;
      if (results[instance] < 0.0) results[instance] = 0.0;
      instance += gridDim.x;
  }
}

__global__ void cosineSimilarityCuda(float3* pDotProducts, size_t pSize, float* results) {
    int instance = blockIdx.x * blockDim.x + threadIdx.x;
  
    while (instance < pSize) {
        results[instance] = pDotProducts[instance].y / (sqrtf(pDotProducts[instance].x)* sqrtf(pDotProducts[instance].z));
        instance += gridDim.x;
    }
}