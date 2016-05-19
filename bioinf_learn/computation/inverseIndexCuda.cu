/**
 Copyright 2016 Joachim Wolff
 Master Thesis
 Tutors: Fabrizio Costa, Milad Miladi
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwigs-University Freiburg im Breisgau
**/

#include "inverseIndexCuda.h"
#include "kernel.h"

InverseIndexCuda::InverseIndexCuda(size_t pNumberOfHashFunctions, 
                                    size_t pShingle, size_t pShingleSize, 
                                    size_t pBlockSize, size_t pHashAlgorithm) {
    mNumberOfHashFunctions = pNumberOfHashFunctions;
    mShingle = pShingle;
    mShingleSize = pShingleSize;
    mBlockSize = pBlockSize;
    mHashAlgorithm = pHashAlgorithm;
}
InverseIndexCuda::~InverseIndexCuda() {
  
}
void InverseIndexCuda::copyDataToGpu(SparseMatrixFloat* pRawData, int** pDevFeatureList,
                                      float** pDevValueList, size_t** pSizeList) {

    // memory for the number of features per instance
    cudaMalloc((void **) &(*pSizeList),
           sizeof(size_t) * pRawData->size());
    // copy the size of all instances to the gpu               
    cudaMemcpy((*pSizeList), pRawData->getSparseMatrixSizeOfInstances(),
            sizeof(size_t) * pRawData->size(),
            cudaMemcpyHostToDevice);
    
    // memory for instances and their featureIds
    cudaMalloc((void **) &(*pDevFeatureList),
            pRawData->size() * pRawData->getMaxNnz() * sizeof(int));
    // memory for the values of the features of the instances
    cudaMalloc((void **) &(*pDevValueList), 
                pRawData->size() * pRawData->getMaxNnz() * sizeof(float));
    
    // copy instances and their feature ids to the gpu
    cudaMemcpy((*pDevFeatureList), pRawData->getSparseMatrixIndex(),
                pRawData->size() * pRawData->getMaxNnz() * sizeof(int),
            cudaMemcpyHostToDevice);
    
    cudaMemcpy((*pDevValueList), pRawData->getSparseMatrixValues(),
                pRawData->size() * pRawData->getMaxNnz() * sizeof(float),
            cudaMemcpyHostToDevice);
    // printf("mDev_FeatureListCOPY, %u\n", (*pDevFeatureList));
    // printf("&mDev_FeatureListCOPY, %u\n", &(*pDevFeatureList));
    // printf("mDev_FeatureListCOPY, %u\n", (*pDevValueList));
    // printf("&mDev_FeatureListCOPY, %u\n", &(*pDevValueList));
    // printf("mDev_FeatureListCOPY, %u\n", (*pSizeList));
    // printf("&mDev_FeatureListCOPY, %u\n", &(*pSizeList));
}
void InverseIndexCuda::computeSignaturesFittingOnGpu(SparseMatrixFloat* pRawData, 
                                                size_t pStartIndex, size_t pEndIndex, 
                                                size_t pNumberOfInstances, size_t pNumberOfBlocks, 
                                                size_t pNumberOfThreads, size_t pShingleFactor, 
                                                size_t pBlockSizeShingle,
                                                vvsize_t_p* pSignatures, size_t pRangeK) {
    // copy data to gpu
    
    copyDataToGpu(pRawData, &mDev_FeatureList, &mDev_ValuesList, &mDev_SizeOfInstanceList);  
                                               
    size_t signaturesSize = ceil(mNumberOfHashFunctions * pBlockSizeShingle / (float) pShingleFactor);
   
    int* instancesHashValues = (int*) malloc(pRawData->size() * signaturesSize * sizeof(int));
    
    // memory for the inverse index on the gpu.
    // for each instance the number of hash functions
    cudaMalloc((void **) &mDev_ComputedSignaturesPerInstance,
            pRawData->size() * signaturesSize * sizeof(int));
    int* dev_SignaturesBlockSize;
    cudaMalloc((void **) &dev_SignaturesBlockSize,
           128 * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(int));
     
    // execute kernel on gpu
    if (mHashAlgorithm == 0) {
        fitCudaMinHash<<<128, 128>>>
        (mDev_FeatureList, 
        mDev_SizeOfInstanceList,  
        mNumberOfHashFunctions, 
        pRawData->getMaxNnz(),
        mDev_ComputedSignaturesPerInstance, 
        pRawData->size(), 0, mBlockSize, mShingleSize, dev_SignaturesBlockSize);
        cudaDeviceSynchronize();
    } else {
        // fitCudaWtaHash<<<128, 128>>>
        // (mDev_FeatureList, 
        // mDev_SizeOfInstanceList,  
        // mNumberOfHashFunctions, 
        // mDev_JumpLength,
        //         mDev_ComputedSignaturesPerInstance, 
        //         end, start, mBlockSize, mShingleSize, dev_SignaturesBlockSize);
    }
                                    
    // copy results back to host  
    cudaMemcpy(instancesHashValues, mDev_ComputedSignaturesPerInstance, 
                pRawData->size() * signaturesSize * sizeof(int),
                cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
                    
    // copy values into one vector per instance
    for(size_t i = 0; i < pRawData->size(); ++i) {
        vsize_t* instance = new vsize_t(signaturesSize);
        for (size_t j = 0; j < signaturesSize; ++j) {
            (*instance)[j] = static_cast<size_t> (instancesHashValues[i*signaturesSize + j]);
        }
        (*pSignatures)[i] = instance;
    }
    
    cudaFree(mDev_ComputedSignaturesPerInstance);
    cudaFree(dev_SignaturesBlockSize);
    cudaFree(mDev_FeatureList);
    cudaFree(mDev_ValuesList);
    cudaFree(mDev_SizeOfInstanceList);
}
void InverseIndexCuda::computeSignaturesQueryOnGpu(SparseMatrixFloat* pRawData, 
                                                size_t pStartIndex, size_t pEndIndex, 
                                                size_t pNumberOfInstances, size_t pNumberOfBlocks, 
                                                size_t pNumberOfThreads, size_t pShingleFactor, 
                                                size_t pBlockSizeShingle,
                                                vvsize_t_p* pSignatures, size_t pRangeK) {
      // copy data to gpu
    int* featureList;
    float* valueList;
    size_t* sizeOfInstances;
    copyDataToGpu(pRawData, &featureList, &valueList, &sizeOfInstances);  
                                               
    size_t signaturesSize = ceil(mNumberOfHashFunctions * pBlockSizeShingle / (float) pShingleFactor);
   
    int* instancesHashValues = (int*) malloc(pRawData->size() * signaturesSize * sizeof(int));
    int* computedSignaturesPerInstance;
    // memory for the inverse index on the gpu.
    // for each instance the number of hash functions
    cudaMalloc((void **) &computedSignaturesPerInstance,
            pRawData->size() * signaturesSize * sizeof(int));
    int* dev_SignaturesBlockSize;
    cudaMalloc((void **) &dev_SignaturesBlockSize,
           128 * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(int));
     
    // execute kernel on gpu
    if (mHashAlgorithm == 0) {
        fitCudaMinHash<<<128, 128>>>
        (featureList, 
        sizeOfInstances,  
        mNumberOfHashFunctions, 
        pRawData->getMaxNnz(),
        computedSignaturesPerInstance, 
        pRawData->size(), 0, mBlockSize, mShingleSize, dev_SignaturesBlockSize);
        cudaDeviceSynchronize();
    } else {
        // fitCudaWtaHash<<<128, 128>>>
        // (featureList, 
        // valueList,
        // sizeOfInstances,  
        // mNumberOfHashFunctions, 
        // pRawData->getMaxNnz(),
        // computedSignaturesPerInstance, 
        // pRawData->size(), 0, mBlockSize, mShingleSize, dev_SignaturesBlockSize, (int) pRangeK);
        // cudaDeviceSynchronize();
    }
                                        
    // copy results back to host  
    cudaMemcpy(instancesHashValues, computedSignaturesPerInstance, 
                pRawData->size() * signaturesSize * sizeof(int),
                cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
                    
    // copy values into one vector per instance
    for(size_t i = 0; i < pRawData->size(); ++i) {
        vsize_t* instance = new vsize_t(signaturesSize);
        for (size_t j = 0; j < signaturesSize; ++j) {
            (*instance)[j] = static_cast<size_t> (instancesHashValues[i*signaturesSize + j]);
        }
        (*pSignatures)[i] = instance;
    }
    
    cudaFree(computedSignaturesPerInstance);
    cudaFree(dev_SignaturesBlockSize);
    cudaFree(featureList);
    cudaFree(valueList);
    cudaFree(sizeOfInstances);
}