#ifndef KERNEL_CUDA
#define KERNEL_CUDA
#define A 0.414213562 // sqrt(2) - 1

__device__ size_t computeHashValueGpu(size_t key, size_t aModulo);
__global__ void fitGpu(const size_t* pFeatureIdList, const size_t* pSizeOfInstanceList,
                    const size_t pNumberOfHashFunctions, const size_t pMaxNnz,
                    size_t* pComputedSignatures, 
                    const size_t pNumberOfInstances);
__global__ void queryGpu(size_t** pSignature, size_t** pInverseIndex, size_t* pElementsPerHashFunction, size_t pNumberOfHashFunctions,
                        size_t* pHitsToBeReturned);
__global__ void euclidianDistanceGpu(size_t* pFeatureIds, size_t* pSizeOfInstanceList,
                                    float* pFeatureValues, size_t pMaxNnz,
                                    size_t* pPossibleInstances, size_t* pSizePerInstance, size_t pMaxCandidates,                             
                                    size_t* pHitsToBeReturned, float* pValuesToBeReturned);
__global__ void cosineSimilarityGpu();
#endif // KERNEL_CUDA