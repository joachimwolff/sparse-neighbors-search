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

__global__ void fitCuda(const int* pFeatureIdList, const int* pSizeOfInstanceList,
                    const int pNumberOfHashFunctions, const int pMaxNnz,
                    int* pComputedSignatures, 
                    const int pNumberOfInstances, const int pStartInstance, 
                    const int pBlockSize, const int pShingleSize,
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
    // printf("pMaxNNz: %i\n", pMaxNnz);
    // if (blockIdx.x == 0 && threadIdx.x == 0) {
    //     printf("InstancesGPUFITTING: ");
    //     for (int i = 0; i < pMaxNnz; ++i) {
    //         // if (i % 100 == 0) {
    //             // for (int j = 0; j < pSizeOfCandidates[i]; ++j) {
    //                 // if (j % 20 == 0) {
    //                     printf ("%f, ", pFeatureIdList[i]);
                        
    //                 // }
    //             // }
    //         // }   
    //     }
    //     // numberOfFeaturesInstance = pSizeOfInstanceList[instanceId];
    // }
    while (instanceId < pNumberOfInstances) {
        // compute the nearestNeighborsValue for every hash function
        // if pBlockSize is greater as 1, hash functions * pBlockSize values 
        // are computed. They will be merged together by a factor of pShingleSize
        // if (threadIdx.x == 0) {
        //     printf ("instanceId: %i, size: %i\n", instanceId, pSizeOfInstanceList[instanceId]);
        // }
        sizeOfInstance = pSizeOfInstanceList[instanceId];
        while (hashFunctionId < pNumberOfHashFunctions * pBlockSize && featureId < pNumberOfInstances*pMaxNnz) {
            for (unsigned int i = 0; i < sizeOfInstance; ++i) {
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
            for (unsigned int i = 1; i < pShingleSize && hashFunctionId+i < pNumberOfHashFunctions * pBlockSize; ++i) {
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

__device__ void sortDesc(cudaInstanceVector* pCandidates, int pInstanceId, int pSize) {
    
    int threadId = threadIdx.x;
    int instance_tmp;
    float value_tmp;
    for (int i = 0; i < pSize / 2; ++i) {
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
__device__ void sortAsc(cudaInstanceVector* pCandidates, int pInstanceId, int pSize) {
    
    int threadId = threadIdx.x;
    int instance_tmp;
    float value_tmp;
    for (int i = 0; i < pSize / 2; ++i) {
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

__global__ void euclideanDistanceCuda(cudaInstanceVector* candidates, int pSize,
                                        int* pSizeOfCandidates,
                                        int* pFeatureList, float* pValuesList,
                                        int* pSizeOfInstanceList, int pMaxNnz) {
    int instanceIdCandidates = blockIdx.x;
    int threadId = threadIdx.x;

    int pointerToFeatureInstance, pointerToFeatureNeighbor, queryIndexInstance,
        queryIndexNeighbor, instanceIdNeighbor,
        indexSparseMatrixNeighbor;
        // featureIdNeighbor, featureIdInstance;
    // bool endOfInstanceNotReached, endOfNeighborNotReached;
    __syncthreads();
    // if (blockIdx.x == 0 && threadIdx.x == 0 && instanceIdCandidates == 0) {
    //     for (int i = 0; i < pMaxNnz * 4337; ++i) {
    //         // if (i % 100 == 0) {
    //             // for (int j = 0; j < pSizeOfCandidates[i]; ++j) {
    //                 // if (j % 20 == 0) {
    //                     printf ("306feature: %i, value: %f\n", pFeatureList[i], pValuesList[i]);
                        
    //                 // }
    //             // }
    //         // }   
    //     }
    //     // numberOfFeaturesInstance = pSizeOfInstanceList[instanceId];
    // }
    __syncthreads();
    // return;
    // __syncthreads();
    // return;
    const int threads = 96;
    __shared__ float euclideanDistance[threads];
    __shared__ float value[threads];
    __shared__ int featureIdNeighbor[threads];
    __shared__ int featureIdInstance[threads];
    __shared__ bool endOfInstanceNotReached[threads];
    __shared__ bool endOfNeighborNotReached[threads];
    __shared__ int numberOfFeaturesInstance;
    __shared__ int numberOfFeaturesNeighbor[threads];
    __shared__ int instanceId;
    __shared__ int indexSparseMatrixInstance;
    // printf("pSize: %i ", pSize);
    
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
        if (threadIdx.x == 0) {
            instanceId = candidates[instanceIdCandidates].instance[0].x;
        }
        // instanceId = pHitsPerQueryInstance[queryIndexInstance].y;
        // instanceIdNeighbor = pHitsPerQueryInstance[queryIndexNeighbor].y;
        
        // get the index positons for the two instances in the sparse matrix
        
        if (threadIdx.x == 0) {
            indexSparseMatrixInstance = instanceId*pMaxNnz;
            numberOfFeaturesInstance = pSizeOfInstanceList[instanceId];
        }
        
        // get the number of features for every instance
       
        
        
        endOfInstanceNotReached[threadIdx.x] = pointerToFeatureInstance < numberOfFeaturesInstance;
        
        
        // __syncthreads();
        // printf("threadId: %i", threadId);
        // __syncthreads();
        
        while (threadId < pSizeOfCandidates[instanceIdCandidates]) {
            instanceIdNeighbor = candidates[instanceIdCandidates].instance[threadId].x;
            indexSparseMatrixNeighbor = instanceIdNeighbor*pMaxNnz;
            numberOfFeaturesNeighbor[threadIdx.x] = pSizeOfInstanceList[instanceIdNeighbor];
            pointerToFeatureInstance = 0;
            pointerToFeatureNeighbor = 0;
            endOfInstanceNotReached[threadIdx.x] = pointerToFeatureInstance < numberOfFeaturesInstance;
            endOfNeighborNotReached[threadIdx.x] = pointerToFeatureNeighbor < numberOfFeaturesNeighbor[threadIdx.x];
            euclideanDistance[threadIdx.x] = 0;
            value[threadIdx.x] = 0;
            
            while (endOfInstanceNotReached[threadIdx.x] && endOfNeighborNotReached[threadIdx.x]) {
                featureIdInstance[threadIdx.x] = pFeatureList[indexSparseMatrixInstance+pointerToFeatureInstance];
                featureIdNeighbor[threadIdx.x] = pFeatureList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor];
                // if(blockIdx.x == 0 && threadIdx.x == 0) {
                    
                    // printf ("instanceSize: %i, neighborSize: %i, featureInstance: %i, featureNeighbor: %i, pointerI: %i, pointerN: %i\n", numberOfFeaturesInstance, numberOfFeaturesNeighbor[threadIdx.x], featureIdInstance[threadIdx.x],  featureIdNeighbor[threadIdx.x], pointerToFeatureInstance, pointerToFeatureNeighbor);
                // }
                if (featureIdInstance[threadIdx.x] == featureIdNeighbor[threadIdx.x]) {
                    // if they are the same substract the values, compute the square and sum it up
                    // printf("valueN: %f, valueI: %f", pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance] , 
                    // pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor]);
                    value[threadIdx.x] = pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance] 
                                    - pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor];
                    
                    printf("11euclid: %f\n", euclideanDistance[threadIdx.x] );
                    // increase both counters to the next element 
                    ++pointerToFeatureInstance;
                    ++pointerToFeatureNeighbor;
                } else if (featureIdInstance[threadIdx.x] < featureIdNeighbor[threadIdx.x]) {
                    // if the feature ids are unequal square only the smaller one and add it to the sum
                    value[threadIdx.x] = pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance];
                    // euclideanDistance[threadIdx.x] += value[threadIdx.x] * value[threadIdx.x];
                    // increase counter for first vector
                    ++pointerToFeatureInstance;
                } else {
                    value[threadIdx.x] = pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor];
                    // euclideanDistance[threadIdx.x] += value[threadIdx.x] * value[threadIdx.x];
                    ++pointerToFeatureNeighbor;
                }
                euclideanDistance[threadIdx.x] += value[threadIdx.x] * value[threadIdx.x];
                endOfInstanceNotReached[threadIdx.x] = pointerToFeatureInstance < numberOfFeaturesInstance;
                endOfNeighborNotReached[threadIdx.x] = pointerToFeatureNeighbor < numberOfFeaturesNeighbor[threadIdx.x];
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
                endOfNeighborNotReached[threadIdx.x] = pointerToFeatureNeighbor < numberOfFeaturesNeighbor[threadIdx.x];
            }
            __syncthreads();
            
            printf("22euclid: %f\n", euclideanDistance[threadIdx.x]);
            // square root of the sum
            // printf("blockId: %i, threadId: %i\n", blockIdx.x, threadIdx.x);
            
            
            // printf("instanceId: %i, neighborId: %i,  euclidean distance: %f, sizeOfCandidates: %i\n",instanceIdCandidates, instanceIdNeighbor, euclideanDistance[threadIdx.x], pSizeOfCandidates[instanceId]);
            // euclideanDistance[threadIdx.x] = sqrtf(euclideanDistance[threadIdx.x]);
            printf("33euclid: %f\n", euclideanDistance[threadIdx.x]);
            
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
        if (threadIdx.x == 0 && blockIdx.x == 0) {
           for (int i = 0; i < pSizeOfCandidates[instanceIdCandidates]; ++i) {
               printf("id: %i, value: %f\n", candidates[instanceIdCandidates].instance[i].x, candidates[instanceIdCandidates].instance[i].y);
           } 
        }
        __syncthreads();
        
        sortDesc(candidates, instanceIdCandidates, pSizeOfCandidates[instanceIdCandidates]);
        __syncthreads();
        if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("SORTED:\n");
            
           for (int i = 0; i < pSizeOfCandidates[instanceIdCandidates]; ++i) {
               printf("id: %i, value: %f\n", candidates[instanceIdCandidates].instance[i].x, candidates[instanceIdCandidates].instance[i].y);
           } 
        }
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     for (int i = 0; i < pSizeOfCandidates[instanceIdCandidates]; ++i) {
        //         printf("candidate: %i, value: %f\n", candidates[instanceIdCandidates].instance[threadId].x, candidates[instanceIdCandidates].instance[threadId].y);
        //     }
        // }
        __syncthreads();
        return;
        instanceIdCandidates += gridDim.x;
        threadId = threadIdx.x;
    }
}

__global__ void cosineSimilarityCuda(cudaInstanceVector* candidates, int pSize,
                                        int* pSizeOfCandidates,
                                        int* pFeatureList, float* pValuesList,
                                        int* pSizeOfInstanceList, int pMaxNnz) {
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