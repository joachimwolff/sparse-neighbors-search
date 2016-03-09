#include "typeDefinitionsBasic.h"

#ifndef KERNEL_CUDA
#define KERNEL_CUDA
#define A 0.414213562 // sqrt(2) - 1

__device__ size_t computeHashValueCuda(size_t key, size_t aModulo);
__global__ void fitCuda(const size_t* pFeatureIdList, const size_t* pSizeOfInstanceList,
                    const size_t pNumberOfHashFunctions, const size_t pMaxNnz,
                    size_t* pComputedSignatures, 
                    const size_t pNumberOfInstances, const size_t pStartInstance,
                    const size_t pBlockSize, const size_t pShingleSize,
                    size_t* pSignaturesBlockSize);
__global__ void createSortedHistogramsCuda(hits* pHitsPerInstance,
                                            const size_t pNumberOfInstances,
                                            int* pHistogram, int2* pRadixSortMemory,
                                            int2* pHistogramSorted, 
                                            size_t* pNumberOfPossibleNeighbors,
                                            size_t pNumberOfNeighbors, size_t pExcessFactor,
                                            size_t* pNeighborhood, float* pDistances, size_t pFast);
__global__ void euclideanDistanceCuda(int2* pHitsPerQueryInstance, size_t* pNumberInstancesToConsider, 
                                        size_t pRangeBetweenInstances, size_t pNumberOfInstances,
                                        size_t* pFeatureList, float* pValuesList,
                                        size_t* pSizeOfInstanceList, size_t pMaxNnz, 
                                        int2* pRadixSortMemory, int pNumberOfNeighbors,
                                        size_t* pNeighborhood, float* pDistances);
__global__ void cosineSimilarityCuda(int2* pHitsPerQueryInstance, size_t* pNumberInstancesToConsider, 
                                        size_t pRangeBetweenInstances, size_t pNumberOfInstances,
                                        size_t* pFeatureList, float* pValuesList,
                                        size_t* pSizeOfInstanceList, size_t pMaxNnz, 
                                        int2* pRadixSortMemory, int pNumberOfNeighbors,
                                        size_t* neighborhood, float* distances);
__device__ void radixSortDesc(size_t pStartPosition, int2* pRadixSortMemory,
                                int2* pSortingMemory, size_t pNumberOfInstances);
__device__ void radixSortAsc(size_t pStartPosition, int2* pRadixSortMemory,
                                int2* pSortingMemory, size_t pNumberOfInstances);
#endif // KERNEL_CUDA