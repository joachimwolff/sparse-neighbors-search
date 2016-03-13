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
                                            histogram* pHistogram, 
                                            // mergeSortingMemory* pMergeSortMemory,
                                            sortedHistogram* pHistogramSorted,
                                            size_t pNneighbors, size_t pFast, size_t pExcessFactor);
__global__ void euclideanDistanceCuda(sortedHistogram* pSortedHistogram, size_t pSizeSortedHistogram,
                                        // mergeSortingMemory* pMergeSortMemory,
                                        size_t* pFeatureList, float* pValuesList,
                                        size_t* pSizeOfInstanceList, size_t pMaxNnz, 
                                        // size_t* pNeighborhood, float* pDistances,
                                        size_t pNneighbors);
__global__ void cosineSimilarityCuda(sortedHistogram* pSortedHistogram, size_t pSizeSortedHistogram,
                                        // mergeSortingMemory* pMergeSortMemory,
                                        size_t* pFeatureList, float* pValuesList,
                                        size_t* pSizeOfInstanceList, size_t pMaxNnz, 
                                        // size_t* pNeighborhood, float* pDistances,
                                        size_t pNneighbors);
__device__ void mergeSortDesc(sortedHistogram* pSortedHistogram, uint pInstanceId);
__device__ void mergeSortAsc(sortedHistogram* pSortedHistogram, uint pInstanceId);
#endif // KERNEL_CUDA