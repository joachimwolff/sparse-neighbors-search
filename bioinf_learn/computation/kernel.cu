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
// #include <cub/cub.cuh>
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
    //  if (threadIdx.x == 0 && blockIdx.x == 0) {
    //         printf("FOO%i\n", __LINE__);
    //     }              
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
        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //     printf("%i\n", sizeOfInstance);
        // }
        while (hashFunctionId < pNumberOfHashFunctions * pBlockSize && featureId < pNumberOfInstances*pMaxNnz) {
            for (size_t i = 0; i < sizeOfInstance; ++i) {
                hashValue = computeHashValueCuda((pFeatureIdList[featureId + i]+1) * (hashFunctionId+1), MAX_VALUE);
                if (hashValue < nearestNeighborsValue) {
                    nearestNeighborsValue = hashValue;
                }
                // if (threadIdx.x == 0 && blockIdx.x == 0) {
                //     printf("%i\n", hashValue);
                // }
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
                if (threadIdx.x == 0 && blockIdx.x == 0) {
                    printf("%i\n", signatureBlockValue);
                }
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
    float __shared__ value[128];
    int __shared__ jumpLength;
    int __shared__ size;
    
    
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
        // threadId = threadIdx.x;
        int i = blockDim.x/2;
        while (i != 0) {
            if (threadIdx.x < i) { 
                value[threadIdx.x] += value[threadIdx.x + i];
            // printf("dotXX: %i, threadId: %i, value: %f\n",instanceId, threadIdx.x, value[threadIdx.x]);
                
            }
            __syncthreads();
            i /= 2;
        }
        
            pDevDotProduct[instanceId] = value[0];
        // if (threadIdx.x == 0) {
        //     printf("dotXZ: %i: %lf\n",instanceId, value[0] / 1000.0);
        // }
        instanceId += gridDim.x;
        threadId = threadIdx.x;
    }                                
}

__device__ float dotProduct(int* pFeatureListX, float* pValuesListX, int pSizeX,
                            int* pFeatureListY, float* pValuesListY, int pSizeY) {
    int counterX = 0;
    int counterY = 0;
    int featureX = pFeatureListX[counterX];
    int featureY = pFeatureListY[counterY];
    float value = 0.0;
    while (counterX < pSizeX && counterY < pSizeY) {
        if (featureX == featureY) {
            value += pValuesListX[counterX] * pValuesListY[counterY];
            ++counterX;
            featureX = pFeatureListX[counterX];
            ++counterY;
            featureY = pFeatureListY[counterY];
        } else if (featureX < featureY) {
            ++counterX;
            featureX = pFeatureListX[counterX];
        } else {
           ++counterY;
           featureY = pFeatureListY[counterY];
        }
    }
    //  if (threadIdx.x == 0) {
    //         printf("%f\n", value);
    //     }
    return value;
}

__device__ float dotProductDevice(int* pFeatureListX, float* pValueListX, 
                                    int pStartPosX, int pEndPosX,
                                    int* pFeatureListY, float* pValueListY, 
                                    int pStartPosY, int pEndPosY) {
    __shared__ int featureIdX[128];
    __shared__ int featureIdY[128];
    __shared__ float value[128];
    int index = 64;
    int round = 0;
    int jumpWidth = 32;
    value[threadIdx.x] = 0.0;
    while (pStartPosX < pEndPosX+(pEndPosX%128) && pStartPosY < pEndPosY+(pEndPosY%128) ) {
        // if (pStartPosX + threadIdx.x < pEndPosX) {
            featureIdX[threadIdx.x] = pFeatureListX[pStartPosX + threadIdx.x];
        // }
        // if (pStartPosY + threadIdx.x < pEndPosY) {
            featureIdY[threadIdx.x] = pFeatureListY[pStartPosY + threadIdx.x];
        // }
        if (featureIdX[0] > featureIdY[127]) {
            pStartPosY += 128;
            continue;
        } else if (featureIdX[127] < featureIdY[0]) {
            pStartPosX += 128;
            continue;
        }
        
        while (round < 128) {
            // if (featureIdX[index] < featureIdY[threadIdx.x]) {
            //     // index -= jumpWidth;
            // } else if (featureIdX[index] > featureIdY[threadIdx.x]) {
            //     index += jumpWidth;
            // } else {
            if (featureIdX[(threadIdx.x+round)%128] == featureIdY[threadIdx.x]) {   
                value[threadIdx.x] += pValueListX[pStartPosX + index] * pValueListY[pStartPosY + threadIdx.x];
                break;
            }
            // jumpWidth /= 2;
            
            ++round;
        }
        __syncthreads();
        // index = 64;
        round = 0;
        if (featureIdX[127] == featureIdY[127]) {
            pStartPosY += 128;
            pStartPosX += 128;
        } else if (featureIdX[127] < featureIdY[127]) {
            pStartPosX += 128;
        } else {
            pStartPosY += 128;
        }
       
    }
     int i = blockDim.x/2;
        while (i != 0) {
            if (threadIdx.x < i) { 
                value[threadIdx.x] += value[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }
        // if (threadIdx.x == 0) {
        //     printf("blockIdx.x: %i, %f\n", blockIdx.x, value[0]);
        // }
        return value[0];
    
}

__global__ void computeDotProducts(float3* pDotProducts, size_t pSize, 
                                        int* pCandidates, size_t* pJumpLength, 
                                        size_t* pCandidateSize,
                                        int* pFeatureIdsNeighbor, float* pValuesNeighbor, 
                                        size_t pMaxNnzNeighbor, size_t* pSizeNeighbor,
                                        int* pFeatureIdsInstance, float* pValuesInstance,
                                        size_t pMaxNnzInstance, size_t* pSizeInstance,
                                         float* pPreComputedDotProductsNeighbor, 
                                         float* pPreComputedDotProductsInstance) {
    int instanceCandidates = blockIdx.x;
    
    __shared__ int instanceCounter;
    __shared__ int neighbor;
    __shared__ int instance;
//     if (threadIdx.x == 0) {
//           printf("FOO %i\n", instanceCandidates);
//   }
    while (instanceCandidates < pSize) {
        if (threadIdx.x == 0) {
            neighbor = pCandidates[pJumpLength[instanceCandidates]];
            // printf("instanceCandidate: %i, neighbor: %i\n", instanceCandidates, neighbor);
            instanceCounter = 0;
        }
        __syncthreads();
        while (instanceCounter < pCandidateSize[neighbor]) {
            if (threadIdx.x == 0) {
                instance = pCandidates[pJumpLength[instanceCandidates]+instanceCounter];
                // printf("instanceCandidate: %i, instance: %i, size: %i\n", instanceCandidates, instance, pCandidateSize[neighbor]);
                
            }
            __syncthreads();
            pDotProducts[pJumpLength[instanceCandidates]+instanceCounter].y = 
                            dotProductDevice(pFeatureIdsNeighbor, pValuesNeighbor, neighbor*pMaxNnzNeighbor, neighbor*pMaxNnzNeighbor + pSizeNeighbor[neighbor],
                                            pFeatureIdsInstance, pValuesInstance, instance*pMaxNnzInstance, instance*pMaxNnzInstance + pSizeInstance[instance]);
            pDotProducts[pJumpLength[instanceCandidates]+instanceCounter].x = pPreComputedDotProductsNeighbor[neighbor];
            pDotProducts[pJumpLength[instanceCandidates]+instanceCounter].z = pPreComputedDotProductsInstance[instance];
            if (threadIdx.x == 0) {
                ++instanceCounter;
               
            }
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                printf("neighbor %i, instance: %i, %f, ", neighbor, instance, pDotProducts[pJumpLength[instanceCandidates]+instanceCounter].y);
                printf(" %f, %f\n", pDotProducts[pJumpLength[instanceCandidates]+instanceCounter].x, pDotProducts[pJumpLength[instanceCandidates]+instanceCounter].z);
             }
        }
        instanceCandidates += gridDim.x;
    }
}
__global__ void euclideanDistanceCuda(float3* pDotProducts, size_t pSize, float* results) {
  int instance = blockIdx.x * blockDim.x + threadIdx.x;
//   if (threadIdx.x == 0) {
//           printf("FOO %i\n", instance);
//   }
  while (instance < pSize) {
      if (threadIdx.x == 0) {
          printf("%f, %f, %f\n",  pDotProducts[instance].x, pDotProducts[instance].y, pDotProducts[instance].z);
      }
      results[instance] = pDotProducts[instance].x - 2*pDotProducts[instance].y + pDotProducts[instance].z;
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