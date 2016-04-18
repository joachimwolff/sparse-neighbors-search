#include "typeDefinitionsBasic.h"
#include "typeDefinitionsCuda.h"
#ifndef KERNEL_CUDA
#define KERNEL_CUDA
#define A 0.414213562 // sqrt(2) - 1

__device__ size_t computeHashValueCuda(int key, int aModulo);
__global__ void fitCudaMinHash(const int* pFeatureIdList, const size_t* pSizeOfInstanceList,
                    const size_t pNumberOfHashFunctions, const size_t pMaxNnz,
                    int* pComputedSignatures, 
                    const size_t pNumberOfInstances, const size_t pStartInstance, 
                    const size_t pBlockSize, const size_t pShingleSize,
                    int* pSignaturesBlockSize);
__global__ void fitCudaWtaHash(const int* pFeatureIdList, const int* pSizeOfInstanceList,
                    const int pNumberOfHashFunctions, const int* pJumpLengthList,
                    int* pComputedSignatures, 
                    const int pNumberOfInstances, const int pStartInstance, 
                    const int pBlockSize, const int pShingleSize,
                    int* pSignaturesBlockSize);
__global__ void euclideanDistanceCuda(cudaInstance* pCandidates, int* pJumpLengthList, 
                                        int* pSizeCandidatesList, int pSize,
                                        int* pFeatureList, float* pValuesList,
                                        int* pSizeOfInstanceList, int* pJumpLength,
                                        float* pDotProduct);
__global__ void cosineSimilarityCuda(cudaInstanceVector* pCandidates, int pSize,
                                        int* pSizeOfCandidates,
                                        int* pFeatureList, float* pValuesList,
                                        int* pSizeOfInstanceList, int* pJumpLength,
                                        float* pDotProduct);
__device__ void sortDesc(cudaInstance* pCandidates, int pInstanceId, int pSize);
__device__ void sortAsc(cudaInstance* pCandidates, int pInstanceId, int pSize);
__global__ void dotProductSingle(int* pFeatureList, float* pValuesList,
                                 size_t* pSizeOfInstanceList,
                                 size_t pSize, size_t pMaxNnz, float* pDevDotProduct);
__device__ float dotProduct(int* pFeatureListX, float* pValuesListX, int pSizeX,
                            int* pFeatureListY, float* pValuesListY, int pSizeY);
#endif // KERNEL_CUDA