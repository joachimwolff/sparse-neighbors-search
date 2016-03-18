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
            pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(int));
    // memory for the values of the features of the instances
    cudaMalloc((void **) &mDev_ValuesList, 
                pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(float));
    // memory for the number of features per instance
    cudaMalloc((void **) &mDev_SizeOfInstanceList,
            pRawData->getNumberOfInstances() * sizeof(int));
    
    int* dev_index = (int*) malloc(sizeof(int) * pRawData->getMaxNnz() * pRawData->getNumberOfInstances());
    for (unsigned int i = 0; i < pRawData->getMaxNnz() * pRawData->getNumberOfInstances(); ++i) {
        dev_index[i] = static_cast<int>(pRawData->getSparseMatrixIndex()[i]);
    }
    // copy instances and their feature ids to the gpu
    cudaMemcpy(mDev_FeatureList, dev_index,
                pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(int),
            cudaMemcpyHostToDevice);
    // copy instances and their values for each feature to the gpu
    float* dev_values = (float*) malloc(sizeof(float) * pRawData->getMaxNnz() * pRawData->getNumberOfInstances());
    for (unsigned int i = 0; i < pRawData->getMaxNnz() * pRawData->getNumberOfInstances(); ++i) {
        dev_values[i] = static_cast<float>(pRawData->getSparseMatrixValues()[i]);
    }
    cudaMemcpy(mDev_ValuesList, dev_values,
                pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(float),
            cudaMemcpyHostToDevice);
            
    int* dev_sizes = (int*) malloc(sizeof(int) * pRawData->getNumberOfInstances());
    for (unsigned int i = 0; i < pRawData->getNumberOfInstances(); ++i) {
        dev_sizes[i] = static_cast<int>(pRawData->getSparseMatrixSizeOfInstances()[i]);
    }
    // copy the size of all instances to the gpu               
    cudaMemcpy(mDev_SizeOfInstanceList, dev_sizes,
            pRawData->getNumberOfInstances() * sizeof(int),
            cudaMemcpyHostToDevice);
     for (unsigned int i = 0; i < pRawData->getNumberOfInstances(); ++i) {
        // printf ("instanceId: %i, size: %i\n", i, dev_sizes[i]);
     }
     
     
    //   for (int i = 0; i < pRawData->getMaxNnz(); ++i) {
    //         // if (i % 100 == 0) {
    //             // for (int j = 0; j < pSizeOfCandidates[i]; ++j) {
    //                 // if (j % 20 == 0) {
    //                     printf ("feature: %i, value: %f\n", dev_index[i], pRawData->getSparseMatrixValues()[i]);
                        
    //                 // }
    //             // }
    //         // }   
    //     }
    free(dev_index);
        free(dev_values);
     free(dev_sizes);
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
    size_t neededMemory = numberOfInstances / iterations  * signaturesSize * sizeof(int);
    neededMemory += pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(int);
    cudaMemGetInfo(&memory_free, &memory_total);
    // do i need more memory than it is free?
    if (neededMemory > memory_free) {
        iterations = ceil(neededMemory / static_cast<float>(memory_free));
    }
    
    size_t start = pStartIndex;
    size_t end = numberOfInstances / iterations;
  
    size_t windowSize = numberOfInstances / iterations;
    int* instancesHashValues = (int*) malloc(numberOfInstances / iterations * mNumberOfHashFunctions * sizeof(int));
    
    // memory for the inverse index on the gpu.
    // for each instance the number of hash functions
    cudaMalloc((void **) &mDev_ComputedSignaturesPerInstance,
            numberOfInstances / iterations  * signaturesSize * sizeof(int));
    int* dev_SignaturesBlockSize;
    cudaMalloc((void **) &dev_SignaturesBlockSize,
           pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(int));
     
    // printf("start: %i, end: %i, iterations: %i\n", start, end, iterations);
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
                    numberOfInstances/iterations * signaturesSize * sizeof(int),
                    cudaMemcpyDeviceToHost);
        // copy values into one vector per instance
        for(size_t i = start; i < end; ++i) {
            vsize_t* instance = new vsize_t(signaturesSize);
            for (size_t j = 0; j < signaturesSize; ++j) {
                (*instance)[j] = static_cast<size_t>(instancesHashValues[i*signaturesSize + j]);
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
    int* dev_featureList;
    int* dev_sizeOfInstanceList;
    int* dev_computedSignaturesPerInstance;
    int numberOfInstances = pEndIndex - pStartIndex;
    int signaturesSize = mNumberOfHashFunctions * pBlockSizeShingle / pShingleFactor;
    
    size_t memory_total = 0;
    size_t memory_free = 0;
    int iterations = 1;
    // memory for all signatures and memory for signatures on each block
    // feature list memory
    int neededMemory = pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(int);
    // memory for the number of features per instance
    neededMemory += pRawData->getNumberOfInstances() * sizeof(int);
    // memory for the signatures per instance
    neededMemory += numberOfInstances / iterations  * signaturesSize * sizeof(int);
    // memory for the signatures per instance for each block before shingle
    neededMemory += pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(int);
    cudaMemGetInfo(&memory_free, &memory_total);
    // do i need more memory than it is free?
    if (neededMemory > memory_free) {
        iterations = ceil(neededMemory / static_cast<float>(memory_free));
    }
    // memory for instances and their featureIds
    cudaMalloc((void **) &dev_featureList,
            pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(int));
    // memory for the number of features per instance
    cudaMalloc((void **) &dev_sizeOfInstanceList,
            pRawData->getNumberOfInstances() * sizeof(int));
    
    // copy instances and their feature ids to the gpu
    int* dev_index = (int*) malloc(sizeof(int) * pRawData->getMaxNnz() * pRawData->getNumberOfInstances());
    for (unsigned int i = 0; i < pRawData->getMaxNnz() * pRawData->getNumberOfInstances(); ++i) {
        dev_index[i] = static_cast<int>(pRawData->getSparseMatrixIndex()[i]);
    }
    cudaMemcpy(dev_featureList, dev_index,
                pRawData->getMaxNnz() * pRawData->getNumberOfInstances() * sizeof(int),
            cudaMemcpyHostToDevice);
    free(dev_index);
    // copy the size of all instances to the gpu    
    int* dev_sizes = (int*) malloc(sizeof(int) * pRawData->getNumberOfInstances());
    for (unsigned int i = 0; i < pRawData->getNumberOfInstances(); ++i) {
        dev_sizes[i] = static_cast<int>(pRawData->getSparseMatrixSizeOfInstances()[i]);
    }           
    cudaMemcpy(dev_sizeOfInstanceList, dev_sizes,
            pRawData->getNumberOfInstances() * sizeof(int),
            cudaMemcpyHostToDevice);
     free(dev_sizes);       
    
    int start = 0;
    int end = numberOfInstances / iterations;
    int windowSize = numberOfInstances / iterations;
    int* instancesHashValues = (int*) malloc(numberOfInstances / iterations * mNumberOfHashFunctions * sizeof(int));
    
    // size_t signaturesSize = mNumberOfHashFunctions * pBlockSizeShingle / pShingleFactor;
    // memory for the signatures on the gpu.
    // for each instance the number of hash functions
    cudaMalloc((void **) &dev_computedSignaturesPerInstance,
            numberOfInstances / iterations  * signaturesSize * sizeof(int));
    int* dev_signaturesBlockSize;
    cudaMalloc((void **) &dev_signaturesBlockSize,
           pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(int));
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
                    numberOfInstances/iterations * signaturesSize * sizeof(int),
                    cudaMemcpyDeviceToHost);
        // copy values into one vector per instance
        for(size_t i = start; i < end; ++i) {
            vsize_t* instance = new vsize_t(signaturesSize);
            for (size_t j = 0; j < signaturesSize; ++j) {
                (*instance)[j] = static_cast<size_t>(instancesHashValues[i*signaturesSize + j]);
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