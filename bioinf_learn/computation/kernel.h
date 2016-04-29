#include "typeDefinitionsBasic.h"
#include "typeDefinitionsCuda.h"
#ifndef KERNEL_CUDA
#define KERNEL_CUDA
#define A 0.414213562 // sqrt(2) - 1

__device__ int computeHashValueCuda(int key, int aModulo);
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
__global__ void euclideanDistanceCuda(float3* pDotProducts, size_t pSize, float* results);
__global__ void cosineSimilarityCuda(float3* pDotProducts, size_t pSize, float* results);
__device__ void sortDesc(cudaInstance* pCandidates, int pInstanceId, int pSize);
__device__ void sortAsc(cudaInstance* pCandidates, int pInstanceId, int pSize);
__global__ void dotProductSingle(int* pFeatureList, float* pValuesList,
                                 size_t* pSizeOfInstanceList,
                                 size_t pSize, size_t pMaxNnz, float* pDevDotProduct);
__device__ float dotProduct(int* pFeatureListX, float* pValuesListX, int pSizeX,
                            int* pFeatureListY, float* pValuesListY, int pSizeY);
__global__ void computeDotProducts(float3* pDotProducts, size_t pSize, 
                                        int* pCandidates, size_t* pJumpLength, 
                                        size_t* pCandidateSize, size_t pNumberOfCandidates,
                                        int* pFeatureIdsNeighbor, float* pValuesNeighbor, 
                                        size_t pMaxNnzNeighbor, size_t* pSizeNeighbor,
                                        int* pFeatureIdsInstance, float* pValuesInstance,
                                        size_t pMaxNnzInstance, size_t* pSizeInstance,
                                         float* pPreComputedDotProductsNeighbor, 
                                         float* pPreComputedDotProductsInstance);
// __global__ void computeDotProducts(size_t* pSizeNeighbor);
#endif // KERNEL_CUDA