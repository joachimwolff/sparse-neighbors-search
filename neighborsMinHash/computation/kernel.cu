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
    size_t minHashValue = MAX_VALUE;
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
        // compute the minHashValue for every hash function
        // if pBlockSize is greater as 1, hash functions * pBlockSize values 
        // are computed. They will be merged together by a factor of pShingleSize
        sizeOfInstance = pSizeOfInstanceList[instanceId];
        while (hashFunctionId < pNumberOfHashFunctions * pBlockSize && featureId < pNumberOfInstances*pMaxNnz) {
            for (size_t i = 0; i < sizeOfInstance; ++i) {
                hashValue = computeHashValueCuda((pFeatureIdList[featureId + i]+1) * (hashFunctionId+1), MAX_VALUE);
                if (hashValue < minHashValue) {
                    minHashValue = hashValue;
                }
            }
            
            pSignaturesBlockSize[signatureBlockId + hashFunctionId] = minHashValue;
            hashFunctionId += blockDim.x;
            minHashValue = MAX_VALUE;
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
        minHashValue = MAX_VALUE;
        hashFunctionId = threadIdx.x;
    }
}



__global__ void queryCuda(size_t* pHitsPerInstance, size_t* pSizePerInstance,
                            size_t pNeighborhoodSize, size_t* pNeighborhood,
                            float* pDistances, const size_t pNumberOfInstances,
                            int* pHistogramMemory, int* pRadixSortMemory,
                            int* pSortedInstancesByNumberOfHits, int* pNumberOfPossibleNeighbors,
                            size_t pNumberOfNeighbors, size_t pExcessFactor) {
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
    int startPositionSortingMemory = blockIdx.x * pNumberOfInstances * 2;
    int addValue = 1;
    size_t numberOfElementsToBeConsidered = pNumberOfNeighbors * pExcessFactor;
    // size_t index;
    // create histogram
    while (instanceId < pNumberOfInstances) {
        for (size_t i = 0; i < pNumberOfInstances; ++i) {
            // clear arrays to 0
            pHistogramMemory[instanceId * pNumberOfInstances + i] = 0;
        }
        // compute start position in array pHitsPerInstance
        startId = instanceId;
        for (size_t i = 0; i < instanceId; ++i) {
            startId += pSizePerInstance[i];
        }
        endId = startId+pSizePerInstance[instanceId];
        
        while (startId + threadId < endId) {
            atomicAdd(&(pHistogramMemory[pHitsPerInstance[startId+threadId] * instanceId]), addValue);
            instanceId += gridDim.x;
            threadId += numberOfThreads;
        }
        
        __syncthreads();
        threadId = threadIdx.x;
        while (threadId < pNumberOfInstances) {
            pSortedInstancesByNumberOfHits[startPositionSortingMemory + threadId] = pHistogramMemory[startId+threadId];
            pSortedInstancesByNumberOfHits[startPositionSortingMemory + threadId + 1] = startId+threadId;
            threadId += blockDim.x;  
        }
        __syncthreads();

        radixSortDesc(startPositionSortingMemory, MAX_VALUE, pRadixSortMemory,
                        pSortedInstancesByNumberOfHits, pNumberOfInstances);
        
        // count number of elements that should be considered in the euclidean distance 
        // or cosine similarity computation
        
        if (pSortedInstancesByNumberOfHits[startPositionSortingMemory + numberOfElementsToBeConsidered*2] != 0) {
            // pSortedInstancesByNumberOfHits[startPositionSortingMemory + numberOfElementsToBeConsidered*2] != 0
        }
    
        instanceId += gridDim.x;
        threadId = threadIdx.x;
    }
}

__device__ void radixSortDesc(int pStartPosition, int pEndPosition, int* pRadixSortMemory,
                            int* pSortingMemory, size_t pNumberOfInstances) {
    // radix sort in descending order of the histogram
    // a[number_of_instances][0] == hits, [1] == elementID
    size_t threadId = threadIdx.x * 2;
    size_t index = 0;
    int addValue = 1;
    size_t bucketNumber = 0;
    __shared__ int elementCount [2];
    for (int i = 0; i < sizeof(int) * 8; ++i) {
        // partion phase: split numbers to bucket 0 or 1
        while (threadId < pNumberOfInstances && threadId < pEndPosition) {
            bucketNumber = (pSortingMemory[pStartPosition+threadId] >> i) & 1;
            atomicAdd(&(elementCount[bucketNumber]), addValue);
            index = pStartPosition+(bucketNumber*pNumberOfInstances) + threadId;
            pRadixSortMemory[index] =  pSortingMemory[pStartPosition + threadId];
            pRadixSortMemory[index+1] =  pSortingMemory[pStartPosition + threadId+1];
            threadId += blockDim.x;
        }
        __syncthreads();
        // collection phase copy values from the bucket 1 and then from bucket 0 to the array
        threadId = threadIdx.x * 2;
        while (threadId < pNumberOfInstances && threadId < pEndPosition) {
            index = pStartPosition + pNumberOfInstances + threadId;
            pSortingMemory[index] = pRadixSortMemory[index];
            pSortingMemory[index+1] = pRadixSortMemory[index+1];
            threadId += blockDim.x;
        }
        
        threadId = threadIdx.x * 2;
        while (threadId < pNumberOfInstances && threadId < pEndPosition) {
            index = pStartPosition + threadId;
            pSortingMemory[index] = pRadixSortMemory[index];
            pSortingMemory[index+1] = pRadixSortMemory[index+1];
            threadId += blockDim.x;
        }
        __syncthreads();
    }
}
__device__ void radixSortAsc(int pStartPosition, int pEndPosition, int* pRadixSortMemory,
                            int* pSortingMemory, size_t pNumberOfInstances) {
    // radix sort in descending order of the histogram
    // a[number_of_instances][0] == hits, [1] == elementID
    size_t threadId = threadIdx.x * 2;
    size_t index = 0;
    int addValue = 1;
    size_t bucketNumber = 0;
    __shared__ int elementCount [2];
    for (int i = 0; i < sizeof(int) * 8; ++i) {
        // partion phase: split numbers to bucket 0 or 1
        while (threadId < pNumberOfInstances && threadId < pEndPosition) {
            bucketNumber = (pSortingMemory[pStartPosition+threadId] >> i) & 1;
            atomicAdd(&(elementCount[bucketNumber]), addValue);
            index = pStartPosition+(bucketNumber*pNumberOfInstances) + threadId;
            pRadixSortMemory[index] =  pSortingMemory[pStartPosition + threadId];
            pRadixSortMemory[index+1] =  pSortingMemory[pStartPosition + threadId+1];
            threadId += blockDim.x;
        }
        __syncthreads();
        // collection phase copy values from the bucket 1 and then from bucket 0 to the array
        threadId = threadIdx.x * 2;
        while (threadId < pNumberOfInstances && threadId < pEndPosition) {
            index = pStartPosition + threadId;
            pSortingMemory[index] = pRadixSortMemory[index];
            pSortingMemory[index+1] = pRadixSortMemory[index+1];
            threadId += blockDim.x;
        }
        
        threadId = threadIdx.x * 2;
        while (threadId < pNumberOfInstances && threadId < pEndPosition) {
            index = pStartPosition + pNumberOfInstances + threadId;
            pSortingMemory[index] = pRadixSortMemory[index];
            pSortingMemory[index+1] = pRadixSortMemory[index+1];
            threadId += blockDim.x;
        }
        __syncthreads();
    }
}
__global__ void euclidianDistanceCuda(int* pHitsPerQueryInstance, int* pNumberInstancesToConsider, 
                                        size_t pRangeBetweenInstances, size_t pNumberOfInstances,
                                        size_t* pFeatureList, float* pValuesList,
                                        size_t* pSizeOfInstanceList, size_t pMaxNnz, 
                                        int* pRadixSortMemory, int pNumberOfNeighbors) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    size_t pointerToFeatureInstance, pointerToFeatureNeighbor, queryIndexInstance,
        queryIndexNeighbor, instanceId, instanceIdNeighbor, indexSparseMatrixInstance,
        indexSparseMatrixNeighbor, numberOfFeaturesInstance, numberOfFeaturesNeighbor,
        featureIdNeighbor, featureIdInstance;
    bool endOfInstanceNotReached, endOfNeighborNotReached;
    float euclideanDistance, value;
    while (blockId < pNumberOfInstances) {
        // pointer to feature ids in sparse matrix
        pointerToFeatureInstance = 0;
        pointerToFeatureNeighbor = 0;
        
        // get the instance ids of the query instance and the possible neighbor
        // it is assumed that the first instance is the query instance and 
        // all others are possible neighbors
        queryIndexInstance = blockId * pRangeBetweenInstances;
        queryIndexNeighbor = blockId * pRangeBetweenInstances + threadId*2;
        
        // get the two instance ids
        instanceId = pHitsPerQueryInstance[queryIndexInstance];
        instanceIdNeighbor = pHitsPerQueryInstance[queryIndexNeighbor];
        
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
        while (threadId < pInstancesToConsider[instanceIdNeighbor]) {
            
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
            pHitsPerQueryInstance[queryIndexNeighbor + 1] = (int) euclideanDistance * 1000;
            threadId += blockIdx.x;
            euclideanDistance = 0;
            value = 0;
            queryIndexNeighbor = blockId * pRangeBetweenInstances + threadId*2;
            instanceIdNeighbor = pHitsPerQueryInstance[queryIndexNeighbor];
            indexSparseMatrixNeighbor = instanceIdNeighbor*pMaxNnz;
            numberOfFeaturesNeighbor = pSizeOfInstanceList[instanceIdNeighbor];
            pointerToFeatureInstance = 0;
            pointerToFeatureNeighbor = 0;
            endOfInstanceNotReached = pointerToFeatureInstance < numberOfFeaturesInstance;
            endOfNeighborNotReached = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
        }
        // sort instances by euclidean distance
        radixSortDesc(queryIndexInstance, pInstancesToConsider[instanceId], 
                        pRadixSortMemory, pHitsPerQueryInstance, pNumberOfInstances);
        blockId += gridDim.x;
        threadId = threadIdx.x;
    }
    
}

__global__ void cosineSimilarityCuda(int* pHitsPerQueryInstance, int* pInstancesToConsider, 
                                        size_t pRangeBetweenInstances, size_t pNumberOfInstances,
                                        size_t* pFeatureList, float* pValuesList,
                                        size_t* pSizeOfInstanceList, size_t pMaxNnz, 
                                        int* pRadixSortMemory, int pNumberOfNeighbors) {
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    size_t pointerToFeatureInstance, pointerToFeatureNeighbor, queryIndexInstance,
        queryIndexNeighbor, instanceId, instanceIdNeighbor, indexSparseMatrixInstance,
        indexSparseMatrixNeighbor, numberOfFeaturesInstance, numberOfFeaturesNeighbor,
        featureIdNeighbor, featureIdInstance;
    bool endOfInstanceNotReached, endOfNeighborNotReached;
    float magnitudeInstance, magnitudeNeighbor, dotProduct, cosineSimilarity;
    while (blockId < pNumberOfInstances) {
        // pointer to feature ids in sparse matrix
        pointerToFeatureInstance = 0;
        pointerToFeatureNeighbor = 0;
        
        // get the instance ids of the query instance and the possible neighbor
        // it is assumed that the first instance is the query instance and 
        // all others are possible neighbors
        queryIndexInstance = blockId * pRangeBetweenInstances;
        queryIndexNeighbor = blockId * pRangeBetweenInstances + threadId*2;
        
        // get the two instance ids
        instanceId = pHitsPerQueryInstance[queryIndexInstance];
        instanceIdNeighbor = pHitsPerQueryInstance[queryIndexNeighbor];
        
        // get the index positons for the two instances in the sparse matrix
        indexSparseMatrixInstance = instanceId*pMaxNnz;
        indexSparseMatrixNeighbor = instanceIdNeighbor*pMaxNnz;
        
        // get the number of features for every instance
        numberOfFeaturesInstance = pSizeOfInstanceList[instanceId];
        numberOfFeaturesNeighbor = pSizeOfInstanceList[instanceIdNeighbor];
        
        endOfInstanceNotReached = pointerToFeatureInstance < numberOfFeaturesInstance;
        endOfNeighborNotReached = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
        magnitudeInstance = 0;
        magnitudeNeighbor = 0;
        dotProduct = 0;
        while (threadId < pInstancesToConsider[instanceIdNeighbor]) {
            
            while (endOfInstanceNotReached && endOfNeighborNotReached) {
                featureIdInstance = pFeatureList[indexSparseMatrixInstance+pointerToFeatureInstance];
                featureIdNeighbor = pFeatureList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor];
                
                if (featureIdInstance == featureIdNeighbor) {
                    // if they are the same substract the values, compute the square and sum it up
                    dotProduct += pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance] 
                                    * pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor];
                    //this->getNextValue(pRowIdVector[i], pointerToMatrixElement) - queryData->getNextValue(pRowId, pointerToVectorElement);
                    magnitudeInstance += powf(pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance], 2);
                    magnitudeNeighbor += powf(pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor], 2);
                    // increase both counters to the next element 
                    ++pointerToFeatureInstance;
                    ++pointerToFeatureNeighbor;
                } else if (featureIdInstance < featureIdNeighbor) {
                    // if the feature ids are unequal square only the smaller one and add it to the sum
                    magnitudeInstance += powf(pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance], 2);
                    // increase counter for first vector
                    ++pointerToFeatureInstance;
                } else {
                    magnitudeNeighbor += powf(pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor], 2);
                    ++pointerToFeatureNeighbor;
                }
                endOfInstanceNotReached = pointerToFeatureInstance < numberOfFeaturesInstance;
                endOfNeighborNotReached = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
            }
            while (endOfInstanceNotReached) {
                magnitudeInstance += powf(pValuesList[indexSparseMatrixInstance+pointerToFeatureInstance], 2);
                ++pointerToFeatureInstance;
                endOfInstanceNotReached = pointerToFeatureInstance < numberOfFeaturesInstance;
            }
            while (endOfNeighborNotReached) {
                magnitudeNeighbor += powf(pValuesList[indexSparseMatrixNeighbor+pointerToFeatureNeighbor], 2);
                ++pointerToFeatureNeighbor;
                endOfNeighborNotReached = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
            }
            
            // square root of the sum
            cosineSimilarity = dotProduct / (float) magnitudeInstance * magnitudeNeighbor;
            // store euclidean distance and neighbor id
            pHitsPerQueryInstance[queryIndexNeighbor + 1] = (int) cosineSimilarity * 1000;
            threadId += blockIdx.x;
            magnitudeInstance = 0;
            magnitudeNeighbor = 0;
            dotProduct = 0;
            cosineSimilarity = 0;
            queryIndexNeighbor = blockId * pRangeBetweenInstances + threadId*2;
            instanceIdNeighbor = pHitsPerQueryInstance[queryIndexNeighbor];
            indexSparseMatrixNeighbor = instanceIdNeighbor*pMaxNnz;
            numberOfFeaturesNeighbor = pSizeOfInstanceList[instanceIdNeighbor];
            pointerToFeatureInstance = 0;
            pointerToFeatureNeighbor = 0;
            endOfInstanceNotReached = pointerToFeatureInstance < numberOfFeaturesInstance;
            endOfNeighborNotReached = pointerToFeatureNeighbor < numberOfFeaturesNeighbor;
        }
        // sort instances by euclidean distance
        radixSortDesc(queryIndexInstance, pInstancesToConsider[instanceId],
                        pRadixSortMemory, pHitsPerQueryInstance, pNumberOfInstances);
        blockId += gridDim.x;
        threadId = threadIdx.x;
    }
    
}