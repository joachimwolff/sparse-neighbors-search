#include "typeDefinitionsBasic.h"
#include "typeDefinitionsCuda.cuh"
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
                                            histogram* pHistogram, radixSortingMemory* pRadixSortMemory,
                                            sortedHistogram* pHistogramSorted, 
                                            // size_t* pNumberOfPossibleNeighbors,
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
__device__ void radixSortDesc(sortedHistogram* pSortedHistogram, radixSortingMemory* pRadixSortMemory);
__device__ void radixSortAsc(sortedHistogram* pSortedHistogram, radixSortingMemory* pRadixSortMemory);
#endif // KERNEL_CUDA