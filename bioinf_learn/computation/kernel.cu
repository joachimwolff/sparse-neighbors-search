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
__device__ size_t computeHashValueCuda(size_t pKey, size_t aModulo) {
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

__global__ void fitCudaMinHash(const size_t* pFeatureIdList, const size_t* pSizeOfInstanceList,
                    const size_t pNumberOfHashFunctions, const size_t pMaxNnz,
                    size_t* pComputedSignatures, 
                    const size_t pNumberOfInstances, const size_t pStartInstance, 
                    const size_t pBlockSize, const size_t pShingleSize,
                    size_t* pSignaturesBlockSize) {
                   
    int instanceId = blockIdx.x + pStartInstance;
    int nearestNeighborsValue = MAX_VALUE;
    int hashValue = 0;
    int signatureSize = pNumberOfHashFunctions * pBlockSize / pShingleSize;
    int featureId;
    int hashFunctionId = threadIdx.x;
    int __shared__ sizeOfInstance;
    int signatureBlockValue;
    int shingleId;
    int signatureBlockId = blockIdx.x * pNumberOfHashFunctions * pBlockSize;
     
    __syncthreads();
    while (instanceId < pNumberOfInstances) {
        // compute the nearestNeighborsValue for every hash function
        // if pBlockSize is greater as 1, hash functions * pBlockSize values 
        // are computed. They will be merged together by a factor of pShingleSize
        if (threadIdx.x == 0) {
            sizeOfInstance = pSizeOfInstanceList[instanceId];
        }
    __syncthreads();
        
        featureId = instanceId * pMaxNnz;
          
        
        while (hashFunctionId < pNumberOfHashFunctions * pBlockSize) {
            for (uint i = 0; i < sizeOfInstance && i < pMaxNnz; ++i) {
                hashValue = computeHashValueCuda((pFeatureIdList[featureId + i]+1) * (hashFunctionId+1), MAX_VALUE);
                if (hashValue < nearestNeighborsValue) {
                    nearestNeighborsValue = hashValue;
                }
            }
            
            pSignaturesBlockSize[signatureBlockId + hashFunctionId] = nearestNeighborsValue;
            hashFunctionId += blockDim.x;
            nearestNeighborsValue = MAX_VALUE;
        }
            // printf("%i\n", __LINE__);

        __syncthreads();
        // merge pShingleSize values together.
        // do one merge per thread
        hashFunctionId = threadIdx.x * pShingleSize;
        shingleId = threadIdx.x;
        while (hashFunctionId < pNumberOfHashFunctions * pBlockSize ) {
            signatureBlockValue = pSignaturesBlockSize[signatureBlockId + hashFunctionId];
            for (unsigned int i = 1; i < pShingleSize && hashFunctionId+i < pNumberOfHashFunctions * pBlockSize; ++i) {
                signatureBlockValue = computeHashValueCuda((pSignaturesBlockSize[signatureBlockId + hashFunctionId+i]+1) * (signatureBlockValue+1), MAX_VALUE);
            }
            pComputedSignatures[(instanceId-pStartInstance)*signatureSize + shingleId] = signatureBlockValue;
            hashFunctionId += blockDim.x * pShingleSize;
            shingleId += blockDim.x;
        }
            // printf("%i\n", __LINE__);

        __syncthreads();
        instanceId += gridDim.x;
        nearestNeighborsValue = MAX_VALUE;
        hashFunctionId = threadIdx.x;
    }
        // printf("%i\n", __LINE__);

}

__global__ void fitCudaWtaHash(const int* pFeatureIdList, const int* pSizeOfInstanceList,
                    const int pNumberOfHashFunctions, const int* pJumpLengthList,
                    int* pComputedSignatures, 
                    const int pNumberOfInstances, const int pStartInstance, 
                    const int pBlockSize, const int pShingleSize,
                    int* pSignaturesBlockSize) {
}


__device__ void sortDesc(cudaInstance* pCandidates, int pInstanceId, int pSize) {
    
    int threadId = 2*threadIdx.x;
    int instance_tmp;
    float value_tmp;
    for (int i = 0; i < pSize / 2; ++i) {
        while (threadId < pSize) {
           
            if (threadId + 1 < pSize 
                    && pCandidates[pInstanceId+threadId+1].y > pCandidates[pInstanceId + threadId].y) {
                        instance_tmp = pCandidates[pInstanceId + threadId].x;
                        value_tmp = pCandidates[pInstanceId + threadId].y;
                        pCandidates[pInstanceId + threadId].x = pCandidates[pInstanceId + threadId + 1].x;
                        pCandidates[pInstanceId + threadId].y = pCandidates[pInstanceId + threadId + 1].y;
                        pCandidates[pInstanceId + threadId + 1].x = instance_tmp;
                        pCandidates[pInstanceId + threadId + 1].y = value_tmp;
            }
            
            if (threadId + 2 < pSize 
                    && pCandidates[pInstanceId + threadId+2].y > pCandidates[pInstanceId + threadId+1].y) {
                        // int2 tmp;
                        instance_tmp = pCandidates[pInstanceId + threadId+1].x;
                        value_tmp = pCandidates[pInstanceId + threadId+1].y;
                        pCandidates[pInstanceId + threadId+1].x = pCandidates[pInstanceId + threadId + 2].x;
                        pCandidates[pInstanceId + threadId+1].y = pCandidates[pInstanceId + threadId + 2].y;
                        pCandidates[pInstanceId + threadId + 2].x = instance_tmp;
                        pCandidates[pInstanceId + threadId + 2].y = value_tmp;
            }
            threadId += blockDim.x;
        }
        __syncthreads();
        threadId = threadIdx.x;
    }
}
__device__ void sortAsc(cudaInstance* pCandidates, int pInstanceId, int pSize) {
    
    int threadId = 2*threadIdx.x;
    int instance_tmp;
    float value_tmp;
    for (int i = 0; i < pSize / 2; ++i) {
        while (threadId < pSize) {
            //  if (blockIdx.x == 0 && pInstanceId == 0 && threadId == 1) {
            //     printf("id: %i, value: %f, id2: %i, value2: %f\n", pCandidates[pInstanceId + threadId].x, pCandidates[pInstanceId + threadId].y,
            //         pCandidates[pInstanceId+threadId+1].x, pCandidates[pInstanceId+threadId+1].y);
            // }
            if (threadId + 1 < pSize 
                    && pCandidates[pInstanceId+threadId+1].y < pCandidates[pInstanceId + threadId].y) {
                        instance_tmp = pCandidates[pInstanceId + threadId].x;
                        value_tmp = pCandidates[pInstanceId + threadId].y;
                        pCandidates[pInstanceId + threadId].x = pCandidates[pInstanceId + threadId + 1].x;
                        pCandidates[pInstanceId + threadId].y = pCandidates[pInstanceId + threadId + 1].y;
                        pCandidates[pInstanceId + threadId + 1].x = instance_tmp;
                        pCandidates[pInstanceId + threadId + 1].y = value_tmp;
            }
            if (threadId + 2 < pSize 
                    && pCandidates[pInstanceId + threadId+2].y < pCandidates[pInstanceId + threadId+1].y) {
                        // int2 tmp;
                        instance_tmp = pCandidates[pInstanceId + threadId+1].x;
                        value_tmp = pCandidates[pInstanceId + threadId+1].y;
                        pCandidates[pInstanceId + threadId+1].x = pCandidates[pInstanceId + threadId + 2].x;
                        pCandidates[pInstanceId + threadId+1].y = pCandidates[pInstanceId + threadId + 2].y;
                        pCandidates[pInstanceId + threadId + 2].x = instance_tmp;
                        pCandidates[pInstanceId + threadId + 2].y = value_tmp;
            }
            __syncthreads();
            threadId += blockDim.x;
        }
        __syncthreads();
        threadId = threadIdx.x;
    }
}
__global__ void dotProductSingle(size_t* pFeatureList, size_t* pValuesList,
                                 size_t* pSizeOfInstanceList, size_t* pJumpLength,
                                 size_t pSize, size_t* pDevDotProduct) {
    int instanceId = blockIdx.x;
    int threadId = threadIdx.x;
    size_t __shared__ value[128];
    int __shared__ jumpLength;
    int __shared__ size;
    
    
    while (instanceId < pSize) {
        value[threadIdx.x] = 0;
        if (threadIdx.x == 0) {
            jumpLength = pJumpLength[instanceId];
            size = pSizeOfInstanceList[instanceId];
        }
        __syncthreads();
        while (threadId < size) {
            value[threadIdx.x] += pValuesList[jumpLength + threadId] * pValuesList[jumpLength + threadId];
            
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
        if (threadIdx.x == 0) {
            pDevDotProduct[instanceId] = value[0];
            // printf("dotXZ: %i: %lf\n",instanceId, (float) value[0]/ (float) 1000000);
        }
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
    int value = 0;
    while (counterX < pSizeX && counterY < pSizeY) {
        if (featureX == featureY) {
            value += (1000* pValuesListX[counterX]) * (1000*pValuesListY[counterY]);
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
    return (float) value / (float) 1000000;
}

// __global__ void dotProduct(int* pFeatureList, int* pValueList, int pStartPosX, int pStartPosY,
//                            int pEndPosX, int pEndPosY) {
//     __shared__ int featureIdX[128];
//     __shared__ int featureIdY[128];
//     __shared__ int value[128];
//     int index = 64;
//     int round = 0;
//     int jumpWidth = 32;
//     value[threadIdx.x] = 0;
//     while (pStartPosX < pEndPosX && pStartPosY < pEndPosY) {
//         if (pStartPosX + threadIdx.x < pEndPosX) {
//             featureIdX[threadId.x] = pFeatureList[pStartPosX + threadIdx.x];
//         } else {
//             featureIdX[threadId.x] = -1;
//         }
//         if (pStartPosY + threadIdx.x < pEndPosY) {
//             featureIdY[threadId.x] = pFeatureList[pStartPosY + threadIdx.x];
//         } else {
//             featureIdY[threadId.x] = -2;
//         }
//         if (featureIdX[0] > featureIdY[128] && featureIdY[128] != -2) {
//             pStartPosX += 128;
//             continue;
//         } else if (featureIdX[128] < featureIdY[0] && featureIdX[128] != -1) {
//             pStartPosY += 128;
//             continue;
//         }
//         while (round < 8) {
//             if (featureIdX[index] < featureIdY[threadIdx.x]) {
//                 index -= jumpWidth;
//             } else if (featureIdX[index] < featureIdY[threadIdx.x]) {
//                 index += jumpWidth;
//             } else {
//                 value[threadIdx.x] += pValueList[pStartPosX + index] * pValueList[pStartPosY + threadIdx.x];
//                 break;
//             }
//             jumpWidth /= 2;
            
//             ++round;
//         }
//         __syncthreads();
//         index = 0;
//         round = 0;
        
       
//     }
//      int i = blockDim.x/2;
//         while (i != 0) {
//             if (threadIdx.x < i) { 
//                 value[threadIdx.x] += value[threadIdx.x + i];
//             // printf("dotXX: %i, threadId: %i, value: %f\n",instanceId, threadIdx.x, value[threadIdx.x]);
                
//             }
//             __syncthreads();
//             i /= 2;
//         }
//         if (threadIdx.x == 0) {
//             pDevDotProduct[instanceId] = (float) value[0] / (float) 1000000.0;
//             // printf("dotXZ: %i: %lf\n",instanceId, (float) value[0]/ (float) 1000000);
//         }
    
// }
__global__ void euclideanDistanceCuda(cudaInstance* pCandidates, int* pJumpLengthList, 
                                        int* pSizeCandidatesList, int pSize,
                                        int* pFeatureList, float* pValuesList,
                                        int* pSizeOfInstanceList, int* pJumpLength,
                                        float* pDotProduct) {
    int instanceIdCandidates = blockIdx.x;
    int threadId = threadIdx.x;
    int candidate;
    // int value;
    int __shared__ size;
    int __shared__ instance;
    float __shared__ value[128];
    float __shared__ dotProductXX;
    // float __shared__ dotProductYY[128];
    while (instanceIdCandidates < pSize) {
        if (threadIdx.x == 0) {
            instance = pCandidates[pJumpLengthList[instanceIdCandidates]].x;
            dotProductXX = pDotProduct[instance];
            size = pSizeCandidatesList[instanceIdCandidates];
            // printf("instance: %i\n", instanceIdCandidates);
        }
        __syncthreads();
        
        while (threadId < size) {
            
            candidate = pCandidates[pJumpLengthList[instanceIdCandidates]+threadId].x;
            // printf("instance: %i, %i, %i, %i, %i\n", instanceIdCandidates, pJumpLengthList[instanceIdCandidates], candidate, size, __LINE__);
            
            // value = candidates[instanceIdCandidates][threadId].y;
            // dotProductYY[threadIdx.x] = pDotProduct[candidate];
            value[threadIdx.x] = dotProduct(&pFeatureList[pJumpLength[instance]], &pValuesList[pJumpLength[instance]], pSizeOfInstanceList[instance],
                                &pFeatureList[pJumpLength[candidate]], &pValuesList[pJumpLength[candidate]],
                                 pSizeOfInstanceList[candidate]);
            // printf("instance: %i, %i\n", instanceIdCandidates, __LINE__);
            
            // if (blockIdx.x == 0 && instanceIdCandidates == 0) {
            //     if (threadIdx.x < size) {
            //         // printf("Bla\n");
            //         // printf("candidate: %i\n", candidate);
            //         printf("value: %f\n", value[threadIdx.x]);
                    
            //         printf("dotYY: %f\n", pDotProduct[candidate]);
                    
                    
            //         // printf("dotXX: %f\n", dotProductXX);
            //         // printf("value: %i\n", candidate);
                    
            //     }
            // }
            // printf("instance__XX: %i, %i, %i, %f\n", instanceIdCandidates, threadIdx.x, __LINE__, dotProductXX);
            
            value[threadIdx.x] *= 2;
            value[threadIdx.x] = dotProductXX - value[threadIdx.x] + pDotProduct[candidate];
            if (value[threadIdx.x] <= 0) {
                value[threadIdx.x] = 0;
            }
            value[threadIdx.x] = sqrtf(value[threadIdx.x]);
            // printf("instance__sqrt: %i, %i, %i, %f\n", instanceIdCandidates, threadIdx.x, __LINE__, value[threadIdx.x]);
            
            pCandidates[pJumpLengthList[instanceIdCandidates]+threadId].y = value[threadIdx.x];
            
            threadId += blockDim.x;
            // printf("instance: %i, %i\n", instanceIdCandidates, __LINE__);
            
        }
        // __syncthreads();
        // // if (blockIdx.x == 0 && instanceIdCandidates == 0) {
        // //     threadId = threadIdx.x;
        // //     if (threadIdx.x == 0) {
        // //         printf("unsorted:\n");
        // //     }
        // // __syncthreads();
        // //     while (threadId < size) {
        // //         printf("threadId, %i, candiate: %i, value: %f\n",threadId, pCandidates[pJumpLengthList[instanceIdCandidates]+threadId].x, pCandidates[pJumpLengthList[instanceIdCandidates]+threadId].y);
                
        // //         threadId += blockDim.x;
        // //     }
        // // }
        // sortAsc(pCandidates, pJumpLengthList[instanceIdCandidates], size); 
        // __syncthreads();
        
        // if (blockIdx.x == 0 && instanceIdCandidates == 0) {
        //     threadId = threadIdx.x;
        //     if (threadIdx.x == 0) {
        //         printf("sorted:\n");
        //     }
        //      __syncthreads();
        //     while (threadId < size) {
        //         printf("threadId, %i, candiate: %i, value: %f\n",threadId, pCandidates[pJumpLengthList[instanceIdCandidates]+threadId].x, pCandidates[pJumpLengthList[instanceIdCandidates]+threadId].y);
        //         threadId += blockDim.x;
        //     }
        // }
        instanceIdCandidates += gridDim.x;
        threadId = threadIdx.x;
    }
    __syncthreads();
}

__global__ void cosineSimilarityCuda(cudaInstanceVector* pCandidates, int pSize,
                                        int* pSizeOfCandidates,
                                        int* pFeatureList, float* pValuesList,
                                        int* pSizeOfInstanceList, int* pJumpLength,
                                        float* pDotProduct) {
    // int blockId = blockIdx.x;
    // int threadId = threadIdx.x;
    // size_t pointerToFeatureInstance, pointerToFeatureNeighbor, queryIndexInstance,
    //     queryIndexNeighbor, instanceId, instanceIdNeighbor, indexSparseMatrixInstance,
    //     indexSparseMatrixNeighbor, numberOfFeaturesInstance, numberOfFeaturesNeighbor,
    //     featureIdNeighbor, featureIdInstance;
    // bool endOfInstanceNotReached, endOfNeighborNotReached;
    // float magnitudeInstance, magnitudeNeighbor, dotProduct, cosineSimilarity;
    // while (blockId < pNumberOfInstances) {
    //     // pointer to feature ids in sparse matrix
    //     pointerToFeatureInstance = 0;
    //     pointerToFeatureNeighbor = 0;
        
    //     // get the instance ids of the query instance and the possible neighbor
    //     // it is assumed that the first instance is the query instance and 
    //     // all others are possible neighbors
    //     queryIndexInstance = blockId * pRangeBetweenInstances;
    //     queryIndexNeighbor = blockId * pRangeBetweenInstances + threadId;
        
    //     // get the two instance ids
    //     instanceId = pHitsPerQueryInstance[queryIndexInstance].y;
    //     instanceIdNeighbor = pHitsPerQueryInstance[queryIndexNeighbor].y;
        
    //     // get the index positons for the two instances in the sparse matrix
    //     indexSparseMatrixInstance = instanceId*pMaxNnz;
    //     indexSparseMatrixNeighbor = instanceIdNeighbor*pMaxNnz;
        
    //     // get the number of features for every instance
    //     numberOfFeaturesInstance = pSizeOfInstanceList[instanceId];
    //     numberOfFeaturesNeighbor = pSizeOfInstanceList[instanceIdNeighbor];
        
    //     endOfInstanceNotReached = pointerToFeatureInstance < numberOfFeaturesInstance;
    //     endOfNeighborNotReached = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
    //     magnitudeInstance = 0;
    //     magnitudeNeighbor = 0;
    //     dotProduct = 0;
    //     while (threadId < pNumberInstancesToConsider[instanceIdNeighbor]) {
            
    //         while (endOfInstanceNotReached && endOfNeighborNotReached) {
    //             featureIdInstance = pFeatureList[indexSparseMatrixInstance+pointerToFeatureInstance];
    //             featureIdNeighbor = pFeatureList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor];
                
    //             if (featureIdInstance == featureIdNeighbor) {
    //                 // if they are the same substract the values, compute the square and sum it up
    //                 dotProduct += pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance] 
    //                                 * pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor];
    //                 //this->getNextValue(pRowIdVector[i], pointerToMatrixElement) - queryData->(pRowId, pointerToVectorElement);
    //                 magnitudeInstance += powf(pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance], 2);
    //                 magnitudeNeighbor += powf(pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor], 2);
    //                 // increase both counters to the next element 
    //                 ++pointerToFeatureInstance;
    //                 ++pointerToFeatureNeighbor;
    //             } else if (featureIdInstance < featureIdNeighbor) {
    //                 // if the feature ids are unequal square only the smaller one and add it to the sum
    //                 magnitudeInstance += powf(pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance], 2);
    //                 // increase counter for first vector
    //                 ++pointerToFeatureInstance;
    //             } else {
    //                 magnitudeNeighbor += powf(pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor], 2);
    //                 ++pointerToFeatureNeighbor;
    //             }
    //             endOfInstanceNotReached = pointerToFeatureInstance < numberOfFeaturesInstance;
    //             endOfNeighborNotReached = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
    //         }
    //         while (endOfInstanceNotReached) {
    //             magnitudeInstance += powf(pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance], 2);
    //             ++pointerToFeatureInstance;
    //             endOfInstanceNotReached = pointerToFeatureInstance < numberOfFeaturesInstance;
    //         }
    //         while (endOfNeighborNotReached) {
    //             magnitudeNeighbor += powf(pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor], 2);
    //             ++pointerToFeatureNeighbor;
    //             endOfNeighborNotReached = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
    //         }
            
    //         // square root of the sum
    //         cosineSimilarity = dotProduct / (float) magnitudeInstance * magnitudeNeighbor;
    //         // store euclidean distance and neighbor id
    //         pHitsPerQueryInstance[queryIndexNeighbor].x = (int) cosineSimilarity * 1000;
    //         threadId += blockIdx.x;
    //         magnitudeInstance = 0;
    //         magnitudeNeighbor = 0;
    //         dotProduct = 0;
    //         cosineSimilarity = 0;
    //         queryIndexNeighbor = blockId * pRangeBetweenInstances + threadId;
    //         instanceIdNeighbor = pHitsPerQueryInstance[queryIndexNeighbor].y;
    //         indexSparseMatrixNeighbor = instanceIdNeighbor*pMaxNnz;
    //         numberOfFeaturesNeighbor = pSizeOfInstanceList[instanceIdNeighbor];
    //         pointerToFeatureInstance = 0;
    //         pointerToFeatureNeighbor = 0;
    //         endOfInstanceNotReached = pointerToFeatureInstance < numberOfFeaturesInstance;
    //         endOfNeighborNotReached = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
    //     }
    //     // sort instances by euclidean distance
    //     mergeSortDesc(queryIndexInstance, pMergeSortMemory, pHitsPerQueryInstance, pNumberOfInstances);
                        
    //     if (threadId < pNumberOfNeighbors) {
    //         pNeighborhood[instanceId*pNumberOfNeighbors+threadId] 
    //             = pHitsPerQueryInstance[queryIndexInstance + threadId].y;
    //         pDistances[instanceId*pNumberOfNeighbors+threadId] 
    //             = (float) pHitsPerQueryInstance[queryIndexInstance + threadId].x;
    //     }
    //     blockId += gridDim.x;
    //     threadId = threadIdx.x;
    // }
    
}