#include "typeDefinitionsBasic.h"
#include "typeDefinitionsCuda.h"
#ifndef KERNEL_CUDA
#define KERNEL_CUDA
#define A 0.414213562 // sqrt(2) - 1

__device__ size_t computeHashValueCuda(size_t key, size_t aModulo);
__global__ void fitCuda(const int* pFeatureIdList, const int* pSizeOfInstanceList,
                    const int pNumberOfHashFunctions, const int pMaxNnz,
                    int* pComputedSignatures, 
                    const int pNumberOfInstances, const int pStartInstance, 
                    const int pBlockSize, const int pShingleSize,
                    int* pSignaturesBlockSize);
__global__ void euclideanDistanceCuda(cudaInstanceVector* candidates, int pSize,
                                        int* pSizeOfCandidates,
                                        int* pFeatureList, float* pValuesList,
                                        int* pSizeOfInstanceList, int pMaxNnz);
__global__ void cosineSimilarityCuda(cudaInstanceVector* candidates, int pSize,
                                        int* pSizeOfCandidates,
                                        int* pFeatureList, float* pValuesList,
                                        int* pSizeOfInstanceList, int pMaxNnz);
__device__ void sortDesc(cudaInstanceVector* pCandidates, int pInstanceId, int pSize);
__device__ void sortAsc(cudaInstanceVector* pCandidates, int pInstanceId, int pSize);
#endif // KERNEL_CUDA