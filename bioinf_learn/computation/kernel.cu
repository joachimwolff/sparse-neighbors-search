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



// __global__ void createSortedHistogramsCuda(hits* pHitsPerInstance,
//                                             const size_t pNumberOfInstances,
//                                             histogram* pHistogram, 
//                                             // mergeSortingMemory* pMergeSortMemory,
//                                             sortedHistogram* pHistogramSorted,
//                                             size_t pNneighbors, size_t pFast, size_t pExcessFactor) {
//     // sort hits per instances
//     // count instances
//     // take highest pNeighborhood*excessfaktor + same hits count
//     // to compute euclidean distance or cosine similarity
    
//     // per block query one instance
//     // sort these with the threads
    
//     int instanceId = blockIdx.x;
//     int threadId = threadIdx.x;
//     int addValue = 1;
//     // int similarValues = 0;
//     // size_t numberOfElementsToBeConsidered = pNumberOfNeighbors * pExcessFactor;
//     __shared__ int elements;
//     __shared__ int sizeOfHistogram;
//     uint index = 0;
//     // size_t index;
//     // create histogram
//     while (instanceId < pNumberOfInstances) {
//         while (threadId < pNumberOfInstances) {
//             // clear arrays to 0
//             pHistogram[blockIdx.x].instances[threadId] = 0;
//             // pHistogramSorted[blockIdx.x].instances[threadId].x = 0;
//             // pHistogramSorted[blockIdx.x].instances[threadId].y = 0;
            
//             threadId += blockDim.x;
//         }
//         __syncthreads();
//         threadId = threadIdx.x;
       
//         while (threadId < pHitsPerInstance[instanceId].size) {
//             atomicAdd(&(pHistogram[blockIdx.x].instances[pHitsPerInstance[instanceId].instances[threadId]]), addValue);
//             threadId += blockDim.x;
//         }
//         __syncthreads();
        
        
//         threadId = threadIdx.x;
//         if (threadIdx.x == 0) {
//             elements = 0;
//         }
//         __syncthreads();
//         while (threadId < pNumberOfInstances) {
//             if (pHistogram[blockIdx.x].instances[threadId] > 0) {
//                 index = atomicAdd(&elements, addValue);
//                 if (index > pNneighbors * pExcessFactor*2) {
//                     // realloc!!!
//                 }
//                 // instance id
//                 pHistogramSorted[instanceId].instances[index].x = threadId;
//                 // number of hits
//                 pHistogramSorted[instanceId].instances[index].y = pHistogram[blockIdx.x].instances[threadId];
                
//             }
//             threadId += blockDim.x;  
//         }
//         __syncthreads(); 
//         if (threadIdx.x == 0) {
//             pHistogramSorted[instanceId].size = elements;
//             elements = 0;
//         }
//         __syncthreads(); 
//         // return;
//         // printf("fast: %i", pFast);
        
//         mergeSortDesc(pHistogramSorted, instanceId);
//         __syncthreads(); 
//         threadId = threadIdx.x;
//         // return;
        
//         // insert the k neighbors and distances to the neighborhood and distances vector
//         // printf("fast: %i", pFast);
//         // if (pFast) {       
//         //     // printf("FAST");
//         //     while (threadId < pNumberOfNeighbors * pExcessFactor && threadId < pHistogramSorted[blockIdx.x].size) {
//         //         // if 
//         //         pNeighborhood[instanceId*pNumberOfNeighbors+threadId] 
//         //             = pHistogramSorted[blockIdx.x].instances[threadId].x;
//         //         pDistances[instanceId*pNumberOfNeighbors+threadId] 
//         //             = (float) pHistogramSorted[blockIdx.x].instances[threadId].y;
//         //         threadId += blockDim.x;
//         //     }
//         //     __syncthreads();
//         // } else {
//             // count number of elements that should be considered in the euclidean distance 
//             // or cosine similarity computation
//             if (threadIdx.x == 0) {
//                 sizeOfHistogram = pHistogramSorted[instanceId].size;
//                 if (pNneighbors * pExcessFactor < sizeOfHistogram) {
//                     elements = pNneighbors * pExcessFactor;
//                     while (elements + 1 < sizeOfHistogram
//                         && pHistogramSorted[instanceId].instances[elements].y > 1
//                         && pHistogramSorted[instanceId].instances[elements].y == pHistogramSorted[instanceId].instances[elements + 1].y) {
//                         ++elements;
//                     }
//                     printf("sizeFFF: %i", elements);
//                     pHistogramSorted[instanceId].size = elements;
//                 }
//             }
//         // }
//         __syncthreads();
        
//         instanceId += gridDim.x;
//         threadId = threadIdx.x;
//     }
// }

__device__ void sortDesc(cudaInstanceVector* pCandidates, uint pInstanceId, uint pSize) {
    
    uint threadId = threadIdx.x;
    uint instance_tmp;
    float value_tmp;
    for (uint i = 0; i < pSize / 2; ++i) {
        while (threadId < pSize) {
            if (threadId + 1 < pSize 
                    && pCandidates[pInstanceId].instance[threadId+1].y > pCandidates[pInstanceId].instance[threadId].y) {
                        instance_tmp = pCandidates[pInstanceId].instance[threadId].x;
                        value_tmp = pCandidates[pInstanceId].instance[threadId].y;
                        pCandidates[pInstanceId].instance[threadId].x = pCandidates[pInstanceId].instance[threadId + 1].x;
                        pCandidates[pInstanceId].instance[threadId].y = pCandidates[pInstanceId].instance[threadId + 1].y;
                        pCandidates[pInstanceId].instance[threadId + 1].x = instance_tmp;
                        pCandidates[pInstanceId].instance[threadId + 1].y = value_tmp;
            }
            if (threadId + 2 < pSize 
                    && pCandidates[pInstanceId].instance[threadId+2].y > pCandidates[pInstanceId].instance[threadId+1].y) {
                        instance_tmp = pCandidates[pInstanceId].instance[threadId+1].x;
                        value_tmp = pCandidates[pInstanceId].instance[threadId+1].y;
                        pCandidates[pInstanceId].instance[threadId+1].x = pCandidates[pInstanceId].instance[threadId + 2].x;
                        pCandidates[pInstanceId].instance[threadId+1].y = pCandidates[pInstanceId].instance[threadId + 2].y;
                        pCandidates[pInstanceId].instance[threadId + 2].x = instance_tmp;
                        pCandidates[pInstanceId].instance[threadId + 2].y = value_tmp;
            }
            __syncthreads();
            threadId += blockDim.x;
        }
        __syncthreads();
        threadId = threadIdx.x;
    }
}
__device__ void sortAsc(cudaInstanceVector* pCandidates, uint pInstanceId, uint pSize) {
    
    uint threadId = threadIdx.x;
    uint instance_tmp;
    float value_tmp;
    for (uint i = 0; i < pSize / 2; ++i) {
        while (threadId < pSize) {
            if (threadId + 1 < pSize 
                    && pCandidates[pInstanceId].instance[threadId+1].y < pCandidates[pInstanceId].instance[threadId].y) {
                        instance_tmp = pCandidates[pInstanceId].instance[threadId].x;
                        value_tmp = pCandidates[pInstanceId].instance[threadId].y;
                        pCandidates[pInstanceId].instance[threadId].x = pCandidates[pInstanceId].instance[threadId + 1].x;
                        pCandidates[pInstanceId].instance[threadId].y = pCandidates[pInstanceId].instance[threadId + 1].y;
                        pCandidates[pInstanceId].instance[threadId + 1].x = instance_tmp;
                        pCandidates[pInstanceId].instance[threadId + 1].y = value_tmp;
            }
            if (threadId + 2 < pSize 
                    && pCandidates[pInstanceId].instance[threadId+2].y < pCandidates[pInstanceId].instance[threadId+1].y) {
                        // int2 tmp;
                        instance_tmp = pCandidates[pInstanceId].instance[threadId+1].x;
                        value_tmp = pCandidates[pInstanceId].instance[threadId+1].y;
                        pCandidates[pInstanceId].instance[threadId+1].x = pCandidates[pInstanceId].instance[threadId + 2].x;
                        pCandidates[pInstanceId].instance[threadId+1].y = pCandidates[pInstanceId].instance[threadId + 2].y;
                        pCandidates[pInstanceId].instance[threadId + 2].x = instance_tmp;
                        pCandidates[pInstanceId].instance[threadId + 2].y = value_tmp;
            }
            __syncthreads();
            threadId += blockDim.x;
        }
        __syncthreads();
        threadId = threadIdx.x;
    }
}

__global__ void euclideanDistanceCuda(cudaInstanceVector* candidates, uint pSize,
                                        uint* pSizeOfCandidates,
                                        size_t* pFeatureList, float* pValuesList,
                                        size_t* pSizeOfInstanceList, size_t pMaxNnz) {
    // int instanceId = blockIdx.x;
    // int threadId = threadIdx.x;
    // int pointerToFeatureInstance, pointerToFeatureNeighbor;
    // // uint pointerToFeatureInstance, pointerToFeatureNeighbor, queryIndexInstance,
    // //     queryIndexNeighbor, instanceIdNeighbor, indexSparseMatrixInstance,
    // //     indexSparseMatrixNeighbor, numberOfFeaturesInstance, numberOfFeaturesNeighbor,
    // //     featureIdInstance, featureIdInstance;
    // __shared__ bool endOfInstanceNotReached[128];
    // __shared__ bool endOfNeighborNotReached[128];
    // // one value per thread
    // __shared__ float euclideanDistance[128];
    // __shared__ float value[128];
    // __shared__ uint numberOfCandidates;
    // // __shared__ uint sizeOfInstanceInSparseMatrix;
    // // __shared__ uint sizeOfNeighborInSparseMatrix;
    
    // __shared__ uint counter_sparseMatrixInstance[128];
    // __shared__ uint counter_sparseMatrixNeighbor[128];
    // __shared__ uint startValue_sparseMatrixInstance;
    // __shared__ uint startValue_sparseMatrixNeighbor[128];
    // __shared__ uint endValue_sparseMatrixInstance;
    // __shared__ uint endValue_sparseMatrixNeighbor[128];
    // __shared__ uint queryIndexInstance;
    // __shared__ uint queryIndexNeighbor[128];
    // __shared__ uint featureIdInstance[128];
    // __shared__ uint featureIdNeighbor[128];
    // while (instanceId < pSize) {
    //     if (threadIdx.x == 0) {
    //         numberOfCandidates = pSizeOfCandidates[instanceId];
    //         // printf("Instance: %i, sizeOfCandidates: %i\n", instanceId, numberOfCandidates);
        
    //         if (numberOfCandidates > 0) {
    //             queryIndexInstance = candidates[instanceId].instance[0].x;
    //             startValue_sparseMatrixInstance = queryIndexInstance*pMaxNnz;
    //             endValue_sparseMatrixInstance = pSizeOfInstanceList[queryIndexInstance];
    //         } else {
    //             instanceId += gridDim.x;
    //             continue;
    //         }
    //         // printf("Instance: %i, instanceReal: %i, sizeOfCandidates: %i\n", instanceId, queryIndexInstance, numberOfCandidates);
    //     }
        
        
    //     // numberOfFeaturesInstance = pSizeOfInstanceList[queryIndexInstance];
    //     // pointerToFeatureInstance = 0;
    //     // pointerToFeatureNeighbor = 0;
    //     __syncthreads();
        
    //     threadId = threadIdx.x;
    //     if (threadIdx.x == 0) {
    //         // printf("\n");
    //     }
    //     while (threadId < numberOfCandidates) {
    //         // init all counters to 0
    //         counter_sparseMatrixInstance[threadIdx.x] = 0;
    //         counter_sparseMatrixNeighbor[threadIdx.x] = 0;
    //         // get real id of neighbor instance
    //         queryIndexNeighbor[threadIdx.x] = candidates[instanceId].instance[threadId].x;
    //         // if (instanceId == 130) {
    //             printf("size: %i, instanceId: %i, threadId: %i, threadIdx.x: %i, queryIndexneighbor: %i part2: %i ",pSizeOfCandidates[instanceId], instanceId, threadId, threadIdx.x, queryIndexNeighbor[threadIdx.x], candidates[instanceId].instance[threadId].x);
    //         // }
    //         // get id in sparse matrix for this neighbor instance
    //         startValue_sparseMatrixNeighbor[threadIdx.x] = queryIndexNeighbor[threadIdx.x]*pMaxNnz;
    //         // get size in sparse matrix for this neighbor instance
            
    //         endValue_sparseMatrixNeighbor[threadIdx.x] = pSizeOfInstanceList[queryIndexNeighbor[threadIdx.x]];
    //         endOfInstanceNotReached[threadIdx.x] = counter_sparseMatrixInstance[threadIdx.x] < endValue_sparseMatrixInstance;
    //         endOfNeighborNotReached[threadIdx.x] = counter_sparseMatrixNeighbor[threadIdx.x] < endValue_sparseMatrixNeighbor[threadIdx.x];
    //         euclideanDistance[threadIdx.x] = 0;
    //         value[threadIdx.x] = 0;
    //         while (endOfInstanceNotReached[threadIdx.x] && endOfNeighborNotReached[threadIdx.x]) {
    //             featureIdInstance[threadIdx.x] = pFeatureList[startValue_sparseMatrixInstance +counter_sparseMatrixInstance[threadIdx.x]];
    //             featureIdNeighbor[threadIdx.x] = pFeatureList[startValue_sparseMatrixNeighbor[threadIdx.x]+counter_sparseMatrixNeighbor[threadIdx.x]];
                
    //             if (featureIdInstance[threadIdx.x] == featureIdNeighbor[threadIdx.x]) {
    //                 // if they are the same substract the values, compute the square and sum it up
    //                 value[threadIdx.x] = pValuesList[startValue_sparseMatrixInstance +counter_sparseMatrixInstance[threadIdx.x]] 
    //                                 - pValuesList[startValue_sparseMatrixNeighbor[threadIdx.x]+counter_sparseMatrixNeighbor[threadIdx.x]];
    //                 euclideanDistance[threadIdx.x] += value[threadIdx.x] * value[threadIdx.x];
    //                 // increase both counters to the next element 
    //                 ++counter_sparseMatrixInstance[threadIdx.x];
    //                 ++counter_sparseMatrixNeighbor[threadIdx.x];
    //             } else if (featureIdInstance[threadIdx.x] < featureIdNeighbor[threadIdx.x]) {
    //                 // if the feature ids are unequal square only the smaller one and add it to the sum
    //                 value[threadIdx.x] = pValuesList[startValue_sparseMatrixInstance +counter_sparseMatrixInstance[threadIdx.x]];
    //                 euclideanDistance[threadIdx.x] += value[threadIdx.x] * value[threadIdx.x];
    //                 // increase counter for first vector
    //                 ++counter_sparseMatrixInstance[threadIdx.x];
    //             } else {
    //                 value[threadIdx.x] = pValuesList[startValue_sparseMatrixNeighbor[threadIdx.x]+counter_sparseMatrixNeighbor[threadIdx.x]];
    //                 euclideanDistance[threadIdx.x] += value[threadIdx.x] * value[threadIdx.x];
    //                 ++counter_sparseMatrixNeighbor[threadIdx.x];
    //             }
    //             endOfInstanceNotReached[threadIdx.x]= counter_sparseMatrixInstance[threadIdx.x] < endValue_sparseMatrixInstance;
    //             endOfNeighborNotReached[threadIdx.x] = counter_sparseMatrixNeighbor[threadIdx.x] < endValue_sparseMatrixNeighbor[threadIdx.x];
    //         }
    //         while (endOfInstanceNotReached[threadIdx.x]) {
    //             value[threadIdx.x] = pValuesList[startValue_sparseMatrixInstance +counter_sparseMatrixInstance[threadIdx.x]];
    //             euclideanDistance[threadIdx.x] += value[threadIdx.x] * value[threadIdx.x];
    //             ++counter_sparseMatrixInstance[threadIdx.x];
    //             endOfInstanceNotReached[threadIdx.x] = counter_sparseMatrixInstance[threadIdx.x] < endValue_sparseMatrixInstance;
    //         }
    //         while (endOfNeighborNotReached[threadIdx.x]) {
    //             value[threadIdx.x] = pValuesList[startValue_sparseMatrixNeighbor[threadIdx.x]+counter_sparseMatrixNeighbor[threadIdx.x]];
    //             euclideanDistance[threadIdx.x] += value[threadIdx.x] * value[threadIdx.x];
    //             ++counter_sparseMatrixNeighbor[threadIdx.x];
    //             endOfNeighborNotReached[threadIdx.x] = counter_sparseMatrixNeighbor[threadIdx.x] < endValue_sparseMatrixNeighbor[threadIdx.x];
    //         }
            
    //         // square root of the sum
    //         // printf("blockId: %i, threadId: %i\n", blockIdx.x, threadIdx.x);
            
    //         // printf("instanceId: %i, neighborId: %i,  threadId: %i, euclidean distance: %f, sizeOfCandidates: %i\n",queryIndexInstance, queryIndexNeighbor, threadId, euclideanDistance, numberOfCandidates);
    //         // return;
    //         candidates[instanceId].instance[threadId].y = sqrtf(euclideanDistance[threadIdx.x]);
    //         // store euclidean distance and neighbor id
    //         // candidates[instanceId].instance[threadId].y =  euclideanDistance;
    //         threadId += blockDim.x;
    //         __syncthreads();
    //     }
    //     __syncthreads();
    //     // sort instances by euclidean distance
    //     sortDesc(candidates, instanceId, pSizeOfCandidates[instanceId]);
    //     // return;
    //     // threadId = threadIdx.x;
    //     __syncthreads();
    //     instanceId += gridDim.x;
    //     threadId = threadIdx.x;
    // }
    int instanceIdCandidates = blockIdx.x;
    int threadId = threadIdx.x;

    size_t pointerToFeatureInstance, pointerToFeatureNeighbor, queryIndexInstance,
        queryIndexNeighbor, instanceIdNeighbor, instanceId, indexSparseMatrixInstance,
        indexSparseMatrixNeighbor, numberOfFeaturesInstance, numberOfFeaturesNeighbor;
        // featureIdNeighbor, featureIdInstance;
    // bool endOfInstanceNotReached, endOfNeighborNotReached;
    const uint threads = 96;
    __shared__ float euclideanDistance[threads];
    __shared__ float value[threads];
    __shared__ int featureIdNeighbor[threads];
    __shared__ int featureIdInstance[threads];
    __shared__ bool endOfInstanceNotReached[threads];
    __shared__ bool endOfNeighborNotReached[threads];
    
    while (instanceIdCandidates < pSize) {
        // pointer to feature ids in sparse matrix
        pointerToFeatureInstance = 0;
        pointerToFeatureNeighbor = 0;
        
        // get the instance ids of the query instance and the possible neighbor
        // it is assumed that the first instance is the query instance and 
        // all others are possible neighbors
        // queryIndexInstance = blockId * pRangeBetweenInstances;
        // queryIndexNeighbor = blockId * pRangeBetweenInstances + threadId;
        
        // get the two instance ids
        // queryIndexNeighbor = threadId + 1;
        instanceId = candidates[instanceIdCandidates].instance[0].x;
        
        // instanceId = pHitsPerQueryInstance[queryIndexInstance].y;
        // instanceIdNeighbor = pHitsPerQueryInstance[queryIndexNeighbor].y;
        
        // get the index positons for the two instances in the sparse matrix
        indexSparseMatrixInstance = instanceId*pMaxNnz;
        
        
        // get the number of features for every instance
        numberOfFeaturesInstance = pSizeOfInstanceList[instanceId];
        
        
        endOfInstanceNotReached[threadIdx.x] = pointerToFeatureInstance < numberOfFeaturesInstance;
        
        
        // __syncthreads();
        // printf("threadId: %i", threadId);
        // __syncthreads();
        
        while (threadId < pSizeOfCandidates[instanceIdCandidates]) {
            instanceIdNeighbor = candidates[instanceIdCandidates].instance[threadId].x;
            indexSparseMatrixNeighbor = instanceIdNeighbor*pMaxNnz;
            numberOfFeaturesNeighbor = pSizeOfInstanceList[instanceIdNeighbor];
            pointerToFeatureInstance = 0;
            pointerToFeatureNeighbor = 0;
            endOfInstanceNotReached[threadIdx.x] = pointerToFeatureInstance < numberOfFeaturesInstance;
            endOfNeighborNotReached[threadIdx.x] = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
            euclideanDistance[threadIdx.x] = 0;
            value[threadIdx.x] = 0;
            
            while (endOfInstanceNotReached[threadIdx.x] && endOfNeighborNotReached[threadIdx.x]) {
                featureIdInstance[threadIdx.x] = pFeatureList[indexSparseMatrixInstance+pointerToFeatureInstance];
                featureIdNeighbor[threadIdx.x] = pFeatureList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor];
                if (featureIdInstance[threadIdx.x] == featureIdNeighbor[threadIdx.x]) {
                    // if they are the same substract the values, compute the square and sum it up
                    value[threadIdx.x] = pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance] 
                                    - pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor];
                    euclideanDistance[threadIdx.x] += value[threadIdx.x] * value[threadIdx.x];
                    // increase both counters to the next element 
                    ++pointerToFeatureInstance;
                    ++pointerToFeatureNeighbor;
                } else if (featureIdInstance[threadIdx.x] < featureIdNeighbor[threadIdx.x]) {
                    // if the feature ids are unequal square only the smaller one and add it to the sum
                    value[threadIdx.x] = pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance];
                    euclideanDistance[threadIdx.x] += value[threadIdx.x] * value[threadIdx.x];
                    // increase counter for first vector
                    ++pointerToFeatureInstance;
                } else {
                    value[threadIdx.x] = pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor];
                    euclideanDistance[threadIdx.x] += value[threadIdx.x] * value[threadIdx.x];
                    ++pointerToFeatureNeighbor;
                }
                endOfInstanceNotReached[threadIdx.x] = pointerToFeatureInstance < numberOfFeaturesInstance;
                endOfNeighborNotReached[threadIdx.x] = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
            }
            while (endOfInstanceNotReached[threadIdx.x]) {
                value[threadIdx.x] = pValuesList[indexSparseMatrixInstance + pointerToFeatureInstance];
                euclideanDistance[threadIdx.x] += value[threadIdx.x] * value[threadIdx.x];
                ++pointerToFeatureInstance;
                endOfInstanceNotReached[threadIdx.x] = pointerToFeatureInstance < numberOfFeaturesInstance;
            }
            while (endOfNeighborNotReached[threadIdx.x]) {
                value[threadIdx.x] = pValuesList[indexSparseMatrixNeighbor + pointerToFeatureNeighbor];
                euclideanDistance[threadIdx.x] += value[threadIdx.x] * value[threadIdx.x];
                ++pointerToFeatureNeighbor;
                endOfNeighborNotReached[threadIdx.x] = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
            }
            
            // square root of the sum
            // printf("blockId: %i, threadId: %i\n", blockIdx.x, threadIdx.x);
            
            // printf("instanceId: %i, neighborId: %i,  euclidean distance: %f, sizeOfCandidates: %i\n",instanceId, queryIndexNeighbor, euclideanDistance, pSizeOfCandidates[instanceId]);
            euclideanDistance[threadIdx.x] = sqrtf(euclideanDistance[threadIdx.x]);
            __syncthreads();
            // printf("threadId: %i, euclidean distance: %f", threadId, euclideanDistance[threadIdx.x]);
            // store euclidean distance and neighbor id
            candidates[instanceIdCandidates].instance[threadId].y =  euclideanDistance[threadIdx.x];
            threadId += blockDim.x;
            
            // instanceIdNeighbor = candidates[instanceId].instance[threadId].x ;
            // indexSparseMatrixNeighbor = instanceIdNeighbor*pMaxNnz;
            // numberOfFeaturesNeighbor = pSizeOfInstanceList[instanceIdNeighbor];
            
        }
        __syncthreads();
        // sort instances by euclidean distance
        sortDesc(candidates, instanceIdCandidates, pSizeOfCandidates[instanceIdCandidates]);
        __syncthreads();
        instanceIdCandidates += gridDim.x;
        threadId = threadIdx.x;
    }
}

__global__ void cosineSimilarityCuda(cudaInstanceVector* candidates, uint pSize,
                                        uint* pSizeOfCandidates,
                                        size_t* pFeatureList, float* pValuesList,
                                        size_t* pSizeOfInstanceList, size_t pMaxNnz) {
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