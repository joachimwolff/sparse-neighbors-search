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
                                    size_t pBlockSize) {
          mNumberOfHashFunctions = pNumberOfHashFunctions;
          mShingle = pShingle;
          mShingleSize = pShingleSize;
          mBlockSize = pBlockSize;
}
InverseIndexCuda::~InverseIndexCuda() {
    cudaFree(mDev_FeatureList);
    cudaFree(mDev_ValuesList);
    cudaFree(mDev_SizeOfInstanceList);
}
void InverseIndexCuda::copyDataToGpu(const SparseMatrixFloat* pRawData) {
    // memory for instances and their featureIds
    cudaMalloc((void **) &mDev_FeatureList,
            pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(size_t));
    // memory for the values of the features of the instances
    cudaMalloc((void **) &mDev_ValuesList, 
                pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(float));
    // memory for the number of features per instance
    cudaMalloc((void **) &mDev_SizeOfInstanceList,
            pRawData->getNumberOfInstances() * sizeof(size_t));
    
    // copy instances and their feature ids to the gpu
    cudaMemcpy(mDev_FeatureList, pRawData->getSparseMatrixIndex(),
                pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(size_t),
            cudaMemcpyHostToDevice);
    // copy instances and their values for each feature to the gpu
    cudaMemcpy(mDev_ValuesList, pRawData->getSparseMatrixValues(),
                pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(float),
            cudaMemcpyHostToDevice);
    // copy the size of all instances to the gpu               
    cudaMemcpy(mDev_SizeOfInstanceList, pRawData->getSparseMatrixSizeOfInstances(),
            pRawData->getNumberOfInstances() * sizeof(size_t),
            cudaMemcpyHostToDevice);
}
vvsize_t_p* InverseIndexCuda::computeSignaturesOnGpu(const SparseMatrixFloat* pRawData, 
                                                        size_t pStartIndex, size_t pEndIndex, 
                                                        size_t pNumberOfInstances,
                                                        size_t pNumberOfBlocks, size_t pNumberOfThreads) {
    std::cout << __LINE__ << std::endl;
    vvsize_t_p* signatures = new vvsize_t_p(pNumberOfInstances);
    // check if enough memory is available on the gpu 
    size_t memory_total = 0;
    size_t memory_free = 0;
    size_t iterations = 1;
    cudaMemGetInfo(&memory_free, &memory_total);
    if (memory_free >= pRawData->getNumberOfInstances()  * mNumberOfHashFunctions * sizeof(size_t)) {
        iterations = ceil(pRawData->getNumberOfInstances()  * mNumberOfHashFunctions * sizeof(size_t) / static_cast<float>(memory_free));
    }
    std::cout << __LINE__ << std::endl;
    
    size_t start = 0;
    size_t end = pRawData->getNumberOfInstances() / iterations;
    size_t windowSize = pRawData->getNumberOfInstances() / iterations;
    size_t* instancesHashValues = (size_t*) malloc(pRawData->getNumberOfInstances() / iterations * mNumberOfHashFunctions * sizeof(size_t));
    
    // memory for the inverse index on the gpu.
    // for each instance the number of hash functions
    cudaMalloc((void **) &mDev_ComputedSignaturesPerInstance,
            pRawData->getNumberOfInstances() / iterations  * mNumberOfHashFunctions * sizeof(size_t));
    std::cout << __LINE__ << std::endl;
    
    for (size_t i = 0; i < iterations; ++i) {
    std::cout << __LINE__ << std::endl;
        
        fitCuda<<<pNumberOfBlocks, pNumberOfThreads, mNumberOfHashFunctions * sizeof(size_t)>>>
        (mDev_FeatureList, 
        mDev_SizeOfInstanceList,  
        mNumberOfHashFunctions, 
        pRawData->getMaxNnz(),
                mDev_ComputedSignaturesPerInstance, 
                end, start);
                
        cudaMemcpy(instancesHashValues, mDev_ComputedSignaturesPerInstance, 
                    pRawData->getNumberOfInstances()/iterations * mNumberOfHashFunctions * sizeof(size_t),
                    cudaMemcpyDeviceToHost);
        for(size_t i = 0; i < pRawData->getNumberOfInstances() / iterations; ++i) {
            vsize_t* instance = new vsize_t(mNumberOfHashFunctions);
            for (size_t j = 0; j < mNumberOfHashFunctions; ++j) {
                (*instance)[j] = instancesHashValues[i*mNumberOfHashFunctions + j];
            }
            signatures->push_back(instance);
        }
        
        start = end+1;
        end = end + windowSize;
    std::cout << __LINE__ << std::endl;
        
    }
    std::cout << __LINE__ << std::endl;
    
    cudaFree(mDev_ComputedSignaturesPerInstance);
    return signatures;
}