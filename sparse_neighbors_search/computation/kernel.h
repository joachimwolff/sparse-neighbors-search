/**
 Copyright 2016, 2017, 2018, 2019, 2020 Joachim Wolff
 PhD Thesis

 Copyright 2015, 2016 Joachim Wolff
 Master Thesis
 Tutor: Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwigs-University Freiburg im Breisgau
**/
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

__global__ void euclideanDistanceCuda(float3* pDotProducts, size_t pSize, float* results);
__global__ void cosineSimilarityCuda(float3* pDotProducts, size_t pSize, float* results);
__device__ void sortDesc(cudaInstance* pCandidates, int pInstanceId, int pSize);
__device__ void sortAsc(cudaInstance* pCandidates, int pInstanceId, int pSize);
__global__ void dotProductSingle(int* pFeatureList, float* pValuesList,
                                 size_t* pSizeOfInstanceList,
                                 size_t pSize, size_t pMaxNnz, float* pDevDotProduct);

__global__ void computeDotProducts(float3* pDotProducts, size_t pSize, 
                                        int* pCandidates, size_t* pJumpLength, 
                                        size_t* pCandidateSize, size_t pNumberOfCandidates,
                                        int* pFeatureIdsNeighbor, float* pValuesNeighbor, 
                                        size_t pMaxNnzNeighbor, size_t* pSizeNeighbor,
                                        int* pFeatureIdsInstance, float* pValuesInstance,
                                        size_t pMaxNnzInstance, size_t* pSizeInstance,
                                         float* pPreComputedDotProductsNeighbor, 
                                         float* pPreComputedDotProductsInstance);
#endif // KERNEL_CUDA