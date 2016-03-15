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
void InverseIndexCuda::copyFittingDataToGpu(const SparseMatrixFloat* pRawData) {
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
void InverseIndexCuda::computeSignaturesFittingOnGpu(const SparseMatrixFloat* pRawData, 
                                                size_t pStartIndex, size_t pEndIndex, 
                                                size_t pNumberOfInstances, size_t pNumberOfBlocks, 
                                                size_t pNumberOfThreads, size_t pShingleFactor, 
                                                size_t pBlockSizeShingle,
                                                vvsize_t_p* pSignatures) {
    // check if enough memory is available on the gpu 
    size_t memory_total = 0;
    size_t memory_free = 0;
    size_t iterations = 1;
    size_t numberOfInstances = pEndIndex - pStartIndex;
    size_t signaturesSize = mNumberOfHashFunctions * pBlockSizeShingle / pShingleFactor;
    
    // memory for all signatures and memory for signatures on each block
    size_t neededMemory = numberOfInstances / iterations  * signaturesSize * sizeof(size_t);
    neededMemory += pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(size_t);
    cudaMemGetInfo(&memory_free, &memory_total);
    // do i need more memory than it is free?
    if (neededMemory > memory_free) {
        iterations = ceil(neededMemory / static_cast<float>(memory_free));
    }
    
    size_t start = pStartIndex;
    size_t end = numberOfInstances / iterations;
  
    size_t windowSize = numberOfInstances / iterations;
    size_t* instancesHashValues = (size_t*) malloc(numberOfInstances / iterations * mNumberOfHashFunctions * sizeof(size_t));
    
    // memory for the inverse index on the gpu.
    // for each instance the number of hash functions
    cudaMalloc((void **) &mDev_ComputedSignaturesPerInstance,
            numberOfInstances / iterations  * signaturesSize * sizeof(size_t));
    size_t* dev_SignaturesBlockSize;
    cudaMalloc((void **) &dev_SignaturesBlockSize,
           pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(size_t));
     
    printf("start: %i, end: %i, iterations: %i\n", start, end, iterations);
    // compute the signatures on the gpu
    // do it in n iterations with equal sized chunks 
    // if the data would not fit on the ram of the gpu
    for (size_t i = 0; i < iterations; ++i) {
        // execute kernel on gpu
        fitCuda<<<pNumberOfBlocks, pNumberOfThreads>>>
        (mDev_FeatureList, 
        mDev_SizeOfInstanceList,  
        mNumberOfHashFunctions, 
        pRawData->getMaxNnz(),
                mDev_ComputedSignaturesPerInstance, 
                end, start, mBlockSize, mShingleSize, dev_SignaturesBlockSize);
        // copy results back to host      
        cudaMemcpy(instancesHashValues, mDev_ComputedSignaturesPerInstance, 
                    numberOfInstances/iterations * signaturesSize * sizeof(size_t),
                    cudaMemcpyDeviceToHost);
        // copy values into one vector per instance
        for(size_t i = start; i < end; ++i) {
            vsize_t* instance = new vsize_t(signaturesSize);
            for (size_t j = 0; j < signaturesSize; ++j) {
                (*instance)[j] = instancesHashValues[i*signaturesSize + j];
            }
            // printf("instance: %i\n", i);

            (*pSignatures)[i] = instance;
        }
        
        start = end+1;
        end = end + windowSize;
    }
    
    cudaFree(mDev_ComputedSignaturesPerInstance);
    cudaFree(dev_SignaturesBlockSize);
}
void InverseIndexCuda::computeSignaturesQueryOnGpu(const SparseMatrixFloat* pRawData, 
                                                size_t pStartIndex, size_t pEndIndex, 
                                                size_t pNumberOfInstances, size_t pNumberOfBlocks, 
                                                size_t pNumberOfThreads, size_t pShingleFactor, 
                                                size_t pBlockSizeShingle,
                                                vvsize_t_p* pSignatures) {
    size_t* dev_featureList;
    size_t* dev_sizeOfInstanceList;
    size_t* dev_computedSignaturesPerInstance;
    size_t numberOfInstances = pEndIndex - pStartIndex;
    size_t signaturesSize = mNumberOfHashFunctions * pBlockSizeShingle / pShingleFactor;
    
    size_t memory_total = 0;
    size_t memory_free = 0;
    size_t iterations = 1;
    // memory for all signatures and memory for signatures on each block
    // feature list memory
    size_t neededMemory = pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(size_t);
    // memory for the number of features per instance
    neededMemory += pRawData->getNumberOfInstances() * sizeof(size_t);
    // memory for the signatures per instance
    neededMemory += numberOfInstances / iterations  * signaturesSize * sizeof(size_t);
    // memory for the signatures per instance for each block before shingle
    neededMemory += pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(size_t);
    cudaMemGetInfo(&memory_free, &memory_total);
    // do i need more memory than it is free?
    if (neededMemory > memory_free) {
        iterations = ceil(neededMemory / static_cast<float>(memory_free));
    }
    // memory for instances and their featureIds
    cudaMalloc((void **) &dev_featureList,
            pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(size_t));
    // memory for the number of features per instance
    cudaMalloc((void **) &dev_sizeOfInstanceList,
            pRawData->getNumberOfInstances() * sizeof(size_t));
    
    // copy instances and their feature ids to the gpu
    cudaMemcpy(dev_featureList, pRawData->getSparseMatrixIndex(),
                pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(size_t),
            cudaMemcpyHostToDevice);
    // copy the size of all instances to the gpu               
    cudaMemcpy(dev_sizeOfInstanceList, pRawData->getSparseMatrixSizeOfInstances(),
            pRawData->getNumberOfInstances() * sizeof(size_t),
            cudaMemcpyHostToDevice);
            
    
    size_t start = 0;
    size_t end = numberOfInstances / iterations;
    size_t windowSize = numberOfInstances / iterations;
    size_t* instancesHashValues = (size_t*) malloc(numberOfInstances / iterations * mNumberOfHashFunctions * sizeof(size_t));
    
    // size_t signaturesSize = mNumberOfHashFunctions * pBlockSizeShingle / pShingleFactor;
    // memory for the signatures on the gpu.
    // for each instance the number of hash functions
    cudaMalloc((void **) &dev_computedSignaturesPerInstance,
            numberOfInstances / iterations  * signaturesSize * sizeof(size_t));
    size_t* dev_signaturesBlockSize;
    cudaMalloc((void **) &dev_signaturesBlockSize,
           pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(size_t));
    // compute the signatures on the gpu
    // do it in n iterations with equal sized chunks 
    // if the data would not fit on the ram of the gpu
    for (size_t i = 0; i < iterations; ++i) {
        // execute kernel on gpu
        fitCuda<<<pNumberOfBlocks, pNumberOfThreads>>>
        (dev_featureList, 
        dev_sizeOfInstanceList,  
        mNumberOfHashFunctions, 
        pRawData->getMaxNnz(),
                dev_computedSignaturesPerInstance, 
                end, start, mBlockSize, mShingleSize, dev_signaturesBlockSize);
        // copy results back to host      
        cudaMemcpy(instancesHashValues, dev_computedSignaturesPerInstance, 
                    numberOfInstances/iterations * signaturesSize * sizeof(size_t),
                    cudaMemcpyDeviceToHost);
        // copy values into one vector per instance
        for(size_t i = start; i < end; ++i) {
            vsize_t* instance = new vsize_t(signaturesSize);
            for (size_t j = 0; j < signaturesSize; ++j) {
                (*instance)[j] = instancesHashValues[i*signaturesSize + j];
            }
            (*pSignatures)[i] = instance;
        }
        
        start = end+1;
        end = end + windowSize;
    }
    
    cudaFree(dev_computedSignaturesPerInstance);
    cudaFree(dev_signaturesBlockSize);
    cudaFree(dev_featureList);       
    cudaFree(dev_computedSignaturesPerInstance);       
           
}