#include "typeDefinitionsBasic.h"
#include "typeDefinitionsCuda.h"
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
__global__ void euclideanDistanceCuda(cudaInstanceVector* candidates, uint pSize,
                                        uint* pSizeOfCandidates,
                                        size_t* pFeatureList, float* pValuesList,
                                        size_t* pSizeOfInstanceList, size_t pMaxNnz);
__global__ void cosineSimilarityCuda(cudaInstanceVector* candidates, uint pSize,
                                        uint* pSizeOfCandidates,
                                        size_t* pFeatureList, float* pValuesList,
                                        size_t* pSizeOfInstanceList, size_t pMaxNnz);
__device__ void sortDesc(cudaInstanceVector* pCandidates, uint pInstanceId, uint pSize);
__device__ void sortAsc(cudaInstanceVector* pCandidates, uint pInstanceId, uint pSize);
#endif // KERNEL_CUDA