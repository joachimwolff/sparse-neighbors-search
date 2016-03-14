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
                    const size_t pBlockSize, const size_t pShingleSize,
                    size_t* pSignaturesBlockSize) {
                        
    int instanceId = blockIdx.x + pStartInstance;
    size_t nearestNeighborsValue = MAX_VALUE;
    size_t hashValue = 0;
    size_t signatureSize = pNumberOfHashFunctions * pBlockSize / pShingleSize;
    int featureId = blockIdx.x * pMaxNnz;
    int hashFunctionId = threadIdx.x;
    size_t sizeOfInstance;
    size_t signatureBlockValue;
    size_t shingleId;
    size_t signatureBlockId = blockIdx.x * pNumberOfHashFunctions * pBlockSize;
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



__global__ void createSortedHistogramsCuda(hits* pHitsPerInstance,
                                            const size_t pNumberOfInstances,
                                            histogram* pHistogram, 
                                            // mergeSortingMemory* pMergeSortMemory,
                                            sortedHistogram* pHistogramSorted,
                                            size_t pNneighbors, size_t pFast, size_t pExcessFactor) {
    // sort hits per instances
    // count instances
    // take highest pNeighborhood*excessfaktor + same hits count
    // to compute euclidean distance or cosine similarity
    
    // per block query one instance
    // sort these with the threads
    
    int instanceId = blockIdx.x;
    int threadId = threadIdx.x;
    int addValue = 1;
    // int similarValues = 0;
    // size_t numberOfElementsToBeConsidered = pNumberOfNeighbors * pExcessFactor;
    __shared__ int elements;
    __shared__ int sizeOfHistogram;
    uint index = 0;
    // size_t index;
    // create histogram
    while (instanceId < pNumberOfInstances) {
        while (threadId < pNumberOfInstances) {
            // clear arrays to 0
            pHistogram[blockIdx.x].instances[threadId] = 0;
            // pHistogramSorted[blockIdx.x].instances[threadId].x = 0;
            // pHistogramSorted[blockIdx.x].instances[threadId].y = 0;
            
            threadId += blockDim.x;
        }
        __syncthreads();
        threadId = threadIdx.x;
       
        while (threadId < pHitsPerInstance[instanceId].size) {
            atomicAdd(&(pHistogram[blockIdx.x].instances[pHitsPerInstance[instanceId].instances[threadId]]), addValue);
            threadId += blockDim.x;
        }
        __syncthreads();
        
        
        threadId = threadIdx.x;
        if (threadIdx.x == 0) {
            elements = 0;
        }
        __syncthreads();
        while (threadId < pNumberOfInstances) {
            if (pHistogram[blockIdx.x].instances[threadId] > 0) {
                index = atomicAdd(&elements, addValue);
                if (index > pNneighbors * pExcessFactor*2) {
                    // realloc!!!
                }
                // instance id
                pHistogramSorted[instanceId].instances[index].x = threadId;
                // number of hits
                pHistogramSorted[instanceId].instances[index].y = pHistogram[blockIdx.x].instances[threadId];
                
            }
            threadId += blockDim.x;  
        }
        __syncthreads(); 
        if (threadIdx.x == 0) {
            pHistogramSorted[instanceId].size = elements;
            elements = 0;
        }
        __syncthreads(); 
        // return;
        // printf("fast: %i", pFast);
        
        mergeSortDesc(pHistogramSorted, instanceId);
        __syncthreads(); 
        threadId = threadIdx.x;
        // return;
        
        // insert the k neighbors and distances to the neighborhood and distances vector
        // printf("fast: %i", pFast);
        // if (pFast) {       
        //     // printf("FAST");
        //     while (threadId < pNumberOfNeighbors * pExcessFactor && threadId < pHistogramSorted[blockIdx.x].size) {
        //         // if 
        //         pNeighborhood[instanceId*pNumberOfNeighbors+threadId] 
        //             = pHistogramSorted[blockIdx.x].instances[threadId].x;
        //         pDistances[instanceId*pNumberOfNeighbors+threadId] 
        //             = (float) pHistogramSorted[blockIdx.x].instances[threadId].y;
        //         threadId += blockDim.x;
        //     }
        //     __syncthreads();
        // } else {
            // count number of elements that should be considered in the euclidean distance 
            // or cosine similarity computation
            if (threadIdx.x == 0) {
                sizeOfHistogram = pHistogramSorted[instanceId].size;
                if (pNneighbors * pExcessFactor < sizeOfHistogram) {
                    elements = pNneighbors * pExcessFactor;
                    while (elements + 1 < sizeOfHistogram
                        && pHistogramSorted[instanceId].instances[elements].y > 1
                        && pHistogramSorted[instanceId].instances[elements].y == pHistogramSorted[instanceId].instances[elements + 1].y) {
                        ++elements;
                    }
                    printf("sizeFFF: %i", elements);
                    pHistogramSorted[instanceId].size = elements;
                }
            }
        // }
        __syncthreads();
        
        instanceId += gridDim.x;
        threadId = threadIdx.x;
    }
}

__device__ void mergeSortDesc(sortedHistogram* pSortedHistogram, uint pInstanceId) {
    
    uint threadId = threadIdx.x;
    for (uint i = 0; i < pSortedHistogram[pInstanceId].size / 2; ++i) {
        while (threadId < pSortedHistogram[pInstanceId].size) {
            if (threadId + 1 < pSortedHistogram[pInstanceId].size 
                    && pSortedHistogram[pInstanceId].instances[threadId+1].y > pSortedHistogram[pInstanceId].instances[threadId].y) {
                        int2 tmp;
                        tmp.x = pSortedHistogram[pInstanceId].instances[threadId].x;
                        tmp.y = pSortedHistogram[pInstanceId].instances[threadId].y;
                        pSortedHistogram[pInstanceId].instances[threadId].x = pSortedHistogram[pInstanceId].instances[threadId + 1].x;
                        pSortedHistogram[pInstanceId].instances[threadId].y = pSortedHistogram[pInstanceId].instances[threadId + 1].y;
                        pSortedHistogram[pInstanceId].instances[threadId + 1].x = tmp.x;
                        pSortedHistogram[pInstanceId].instances[threadId + 1].y = tmp.y;
            }
            if (threadId + 2 < pSortedHistogram[pInstanceId].size 
                    && pSortedHistogram[pInstanceId].instances[threadId+2].y > pSortedHistogram[pInstanceId].instances[threadId+1].y) {
                        int2 tmp;
                        tmp.x = pSortedHistogram[pInstanceId].instances[threadId+1].x;
                        tmp.y = pSortedHistogram[pInstanceId].instances[threadId+1].y;
                        pSortedHistogram[pInstanceId].instances[threadId+1].x = pSortedHistogram[pInstanceId].instances[threadId + 2].x;
                        pSortedHistogram[pInstanceId].instances[threadId+1].y = pSortedHistogram[pInstanceId].instances[threadId + 2].y;
                        pSortedHistogram[pInstanceId].instances[threadId + 2].x = tmp.x;
                        pSortedHistogram[pInstanceId].instances[threadId + 2].y = tmp.y;
            }
            __syncthreads();
            threadId += blockDim.x;
        }
        __syncthreads();
        threadId = threadIdx.x;
    }
    // int iterations = ceilf(log2((float)pSortedHistogram[blockIdx.x].size));
    // int instanceId = blockIdx.x;
    // int threadId = threadIdx.x;
    // int index0 = 0;
    // int index1 = 0;
    // __shared__ int pow;
    // for (size_t i = 0; i < iterations; ++i) {
    //     if (threadIdx.x == 0) {
    //         pow = powf(2, i);
    //     }
    //     __syncthreads();
    //     // printf("pow: %i", pow);
    //     int indexDestination = pow * threadId * 2 ;
    //     index0 = pow * threadId * 2 ;
    //     index1 = pow * threadId * 2 + pow;
    //     while (threadId < pSortedHistogram[instanceId].size 
    //                 && index0 < pSortedHistogram[instanceId].size
    //                 && index1 < pSortedHistogram[instanceId].size) {
                
    //             int j = 0;
    //             int counter0 = 0;
    //             int counter1 = 0;
    //             // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //             //         printf("111j: %i, pow: %i, index0: %i, index1: %i\n", j, pow, index0, index1);
    //             //         printf("\nsize: %i\n", pSortedHistogram[instanceId].size);
    //             //     }
    //             while (j < pow*2 && counter0 < pow && counter1 < pow 
    //                         && index0 < pSortedHistogram[instanceId].size 
    //                         && index1 < pSortedHistogram[instanceId].size) {
    //                 if (blockIdx.x == 0 && threadIdx.x == 0) {
    //                     printf("j: %i, pow: %i, counter0: %i, counter1: %i\n", j, pow, counter0, counter1);
    //                     printf("index0: %i , Index1: %i, size: %i\n", index0, index1, pSortedHistogram[instanceId].size);
    //                 }
    //                 if (pSortedHistogram[instanceId].instances[index0].y > pSortedHistogram[instanceId].instances[index1].y) {
    //                     pMergeSortMemory[blockIdx.x].instances[indexDestination + j].x = pSortedHistogram[instanceId].instances[index0].x;
    //                     pMergeSortMemory[blockIdx.x].instances[indexDestination + j].y = pSorkktedHistogram[instanceId].instances[index0].y;
    //                     ++index0;
    //                     ++counter0;
    //                     ++j;
    //                 } else if (pSortedHistogram[instanceId].instances[index0].y < pSortedHistogram[instanceId].instances[index1].y){
    //                     pMergeSortMemory[blockIdx.x].instances[indexDestination + j].x = pSortedHistogram[instanceId].instances[index1].x;
    //                     pMergeSortMemory[blockIdx.x].instances[indexDestination + j].y = pSortedHistogram[instanceId].instances[index1].y;
    //                     ++index1;
    //                     ++counter1;
    //                     ++j;
    //                 } else {
    //                     pMergeSortMemory[blockIdx.x].instances[indexDestination + j].x = pSortedHistogram[instanceId].instances[index0].x;
    //                     pMergeSortMemory[blockIdx.x].instances[indexDestination + j].y = pSortedHistogram[instanceId].instances[index0].y;
    //                     pMergeSortMemory[blockIdx.x].instances[indexDestination+ j + 1].x = pSortedHistogram[instanceId].instances[index1].x;
    //                     pMergeSortMemory[blockIdx.x].instances[indexDestination + j + 1].y = pSortedHistogram[instanceId].instances[index1].y;
    //                     ++index0;
    //                     ++index1;
    //                     ++counter0;
    //                     ++counter1;
    //                     j += 2;
    //                 }
    //             }
    //             while (j < pow*2 && counter0 < pow && index0 < pSortedHistogram[instanceId].size) {
    //                 pMergeSortMemory[blockIdx.x].instances[indexDestination + j].x = pSortedHistogram[instanceId].instances[index0].x;
    //                 pMergeSortMemory[blockIdx.x].instances[indexDestination + j].y = pSortedHistogram[instanceId].instances[index0].y;
    //                 ++index0;
    //                 ++counter0;
    //                 ++j;
    //             }
    //             while (j < pow*2 && counter1 < pow && index1 < pSortedHistogram[instanceId].size) {
    //                 pMergeSortMemory[blockIdx.x].instances[indexDestination + j].x = pSortedHistogram[instanceId].instances[index1].x;
    //                 pMergeSortMemory[blockIdx.x].instances[indexDestination + j].y = pSortedHistogram[instanceId].instances[index1].y;
    //                 ++index1;
    //                 ++counter1;
    //                 ++j;
    //             }
    //         __syncthreads();
    //         threadId += blockDim.x;
    //         index0 = pow * threadId * 2 ;
    //         index1 = pow * threadId * 2 + pow;
    //     }
    //     __syncthreads();
    //     // return;
    //     threadId = threadIdx.x;
    //     while (threadId < pSortedHistogram[instanceId].size) {
    //         pSortedHistogram[instanceId].instances[threadId].x = pMergeSortMemory[blockIdx.x].instances[threadId].x;
    //         pSortedHistogram[instanceId].instances[threadId].y = pMergeSortMemory[blockIdx.x].instances[threadId].y;
    //         threadId += blockDim.x;
    //     }
    //     // return;
    //     threadId = threadIdx.x;
    // }
}
__device__ void mergeSortAsc(sortedHistogram* pSortedHistogram, uint pInstanceId) {
    
    uint threadId = threadIdx.x;
    for (uint i = 0; i < pSortedHistogram[pInstanceId].size / 2; ++i) {
        while (threadId < pSortedHistogram[pInstanceId].size) {
            if (threadId + 1 < pSortedHistogram[pInstanceId].size 
                    && pSortedHistogram[pInstanceId].instances[threadId+1].y < pSortedHistogram[pInstanceId].instances[threadId].y) {
                        int2 tmp;
                        tmp.x = pSortedHistogram[pInstanceId].instances[threadId].x;
                        tmp.y = pSortedHistogram[pInstanceId].instances[threadId].y;
                        pSortedHistogram[pInstanceId].instances[threadId].x = pSortedHistogram[pInstanceId].instances[threadId + 1].x;
                        pSortedHistogram[pInstanceId].instances[threadId].y = pSortedHistogram[pInstanceId].instances[threadId + 1].y;
                        pSortedHistogram[pInstanceId].instances[threadId + 1].x = tmp.x;
                        pSortedHistogram[pInstanceId].instances[threadId + 1].y = tmp.y;
            }
            if (threadId + 2 < pSortedHistogram[pInstanceId].size 
                    && pSortedHistogram[pInstanceId].instances[threadId+2].y < pSortedHistogram[pInstanceId].instances[threadId+1].y) {
                        int2 tmp;
                        tmp.x = pSortedHistogram[pInstanceId].instances[threadId+1].x;
                        tmp.y = pSortedHistogram[pInstanceId].instances[threadId+1].y;
                        pSortedHistogram[pInstanceId].instances[threadId+1].x = pSortedHistogram[pInstanceId].instances[threadId + 2].x;
                        pSortedHistogram[pInstanceId].instances[threadId+1].y = pSortedHistogram[pInstanceId].instances[threadId + 2].y;
                        pSortedHistogram[pInstanceId].instances[threadId + 2].x = tmp.x;
                        pSortedHistogram[pInstanceId].instances[threadId + 2].y = tmp.y;
            }
            __syncthreads();
            threadId += blockDim.x;
        }
        __syncthreads();
        threadId = threadIdx.x;
    }
}
__global__ void kneighborsExact() {
    
}
__global__ void euclideanDistanceCuda(sortedHistogram* pSortedHistogram, size_t pSizeSortedHistogram,
                                        // mergeSortingMemory* pMergeSortMemory,
                                        size_t* pFeatureList, float* pValuesList,
                                        size_t* pSizeOfInstanceList, size_t pMaxNnz, 
                                        // size_t* pNeighborhood, float* pDistances,
                                        size_t pNneighbors) {
    int instanceId = blockIdx.x;
    int threadId = threadIdx.x;
    size_t pointerToFeatureInstance, pointerToFeatureNeighbor, queryIndexInstance,
        queryIndexNeighbor, instanceIdNeighbor, indexSparseMatrixInstance,
        indexSparseMatrixNeighbor, numberOfFeaturesInstance, numberOfFeaturesNeighbor,
        featureIdNeighbor, featureIdInstance;
    bool endOfInstanceNotReached, endOfNeighborNotReached;
    float euclideanDistance, value;
    
    while (instanceId < pSizeSortedHistogram) {
        // pointer to feature ids in sparse matrix
        pointerToFeatureInstance = 0;
        pointerToFeatureNeighbor = 0;
        
        // get the instance ids of the query instance and the possible neighbor
        // it is assumed that the first instance is the query instance and 
        // all others are possible neighbors
        // queryIndexInstance = blockId * pRangeBetweenInstances;
        // queryIndexNeighbor = blockId * pRangeBetweenInstances + threadId;
        
        // get the two instance ids
        queryIndexNeighbor = threadId + 1;
        instanceId = pSortedHistogram[instanceId].instances[0].x;
        instanceIdNeighbor = pSortedHistogram[instanceId].instances[queryIndexNeighbor].x;
        // instanceId = pHitsPerQueryInstance[queryIndexInstance].y;
        // instanceIdNeighbor = pHitsPerQueryInstance[queryIndexNeighbor].y;
        
        // get the index positons for the two instances in the sparse matrix
        indexSparseMatrixInstance = instanceId*pMaxNnz;
        indexSparseMatrixNeighbor = instanceIdNeighbor*pMaxNnz;
        
        // get the number of features for every instance
        numberOfFeaturesInstance = pSizeOfInstanceList[instanceId];
        numberOfFeaturesNeighbor = pSizeOfInstanceList[instanceIdNeighbor];
        
        endOfInstanceNotReached = pointerToFeatureInstance < numberOfFeaturesInstance;
        endOfNeighborNotReached = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
        euclideanDistance = 0;
        value = 0;
        while (threadId < pSortedHistogram[instanceId].size) {
            
            while (endOfInstanceNotReached && endOfNeighborNotReached) {
                featureIdInstance = pFeatureList[indexSparseMatrixInstance+pointerToFeatureInstance];
                featureIdNeighbor = pFeatureList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor];
                
                if (featureIdInstance == featureIdNeighbor) {
                    // if they are the same substract the values, compute the square and sum it up
                    value = pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance] 
                                    - pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor];
                    //this->getNextValue(pRowIdVector[i], pointerToMatrixElement) - queryData->getNextValue(pRowId, pointerToVectorElement);
                    euclideanDistance += value * value;
                    // increase both counters to the next element 
                    ++pointerToFeatureInstance;
                    ++pointerToFeatureNeighbor;
                } else if (featureIdInstance < featureIdNeighbor) {
                    // if the feature ids are unequal square only the smaller one and add it to the sum
                    value = pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance];
                    euclideanDistance += value * value;
                    // increase counter for first vector
                    ++pointerToFeatureInstance;
                } else {
                    value = pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor];
                    euclideanDistance += value * value;
                    ++pointerToFeatureNeighbor;
                }
                endOfInstanceNotReached = pointerToFeatureInstance < numberOfFeaturesInstance;
                endOfNeighborNotReached = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
            }
            while (endOfInstanceNotReached) {
                value = pValuesList[indexSparseMatrixInstance + pointerToFeatureInstance];
                euclideanDistance += value * value;
                ++pointerToFeatureInstance;
                endOfInstanceNotReached = pointerToFeatureInstance < numberOfFeaturesInstance;
            }
            while (endOfNeighborNotReached) {
                value = pValuesList[indexSparseMatrixNeighbor + pointerToFeatureNeighbor];
                euclideanDistance += value * value;
                ++pointerToFeatureNeighbor;
                endOfNeighborNotReached = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
            }
            
            // square root of the sum
            euclideanDistance = sqrtf(euclideanDistance);
            // store euclidean distance and neighbor id
            pSortedHistogram[instanceId].instances[queryIndexNeighbor].y = (int) euclideanDistance * 1000;
            threadId += blockIdx.x;
            euclideanDistance = 0;
            value = 0;
            queryIndexNeighbor += blockIdx.x;
            instanceIdNeighbor = pSortedHistogram[instanceId].instances[queryIndexNeighbor].x ;
            indexSparseMatrixNeighbor = instanceIdNeighbor*pMaxNnz;
            numberOfFeaturesNeighbor = pSizeOfInstanceList[instanceIdNeighbor];
            pointerToFeatureInstance = 0;
            pointerToFeatureNeighbor = 0;
            endOfInstanceNotReached = pointerToFeatureInstance < numberOfFeaturesInstance;
            endOfNeighborNotReached = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
        }
        __syncthreads();
        // sort instances by euclidean distance
        mergeSortDesc(pSortedHistogram, instanceId);
        threadId = threadIdx.x;
        // insert the k neighbors and distances to the neighborhood and distances vector
        // while (threadId < pNneighbors) {
        //     pNeighborhood[instanceId*pNneighbors+threadId] 
        //         = pSortedHistogram[instanceId].instances[threadId].x;
        //     pDistances[instanceId*pNneighbors+threadId] 
        //         = (float) pSortedHistogram[instanceId].instances[threadId].y / (float) 1000;
        //     threadId += blockDim.x;
        // }
        __syncthreads();
        instanceId += gridDim.x;
        threadId = threadIdx.x;
    }
    
}

__global__ void cosineSimilarityCuda(sortedHistogram* pSortedHistogram, size_t pSizeSortedHistogram,
                                        // mergeSortingMemory* pMergeSortMemory,
                                        size_t* pFeatureList, float* pValuesList,
                                        size_t* pSizeOfInstanceList, size_t pMaxNnz, 
                                        // size_t* pNeighborhood, float* pDistances,
                                        size_t pNneighbors) {
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
    //                 //this->getNextValue(pRowIdVector[i], pointerToMatrixElement) - queryData->getNextValue(pRowId, pointerToVectorElement);
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