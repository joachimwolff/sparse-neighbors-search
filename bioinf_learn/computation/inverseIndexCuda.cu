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

InverseIndexCuda::InverseIndexCuda(int pNumberOfHashFunctions, 
                                    int pShingle, int pShingleSize, 
                                    int pBlockSize, int pHashAlgorithm) {
    mNumberOfHashFunctions = pNumberOfHashFunctions;
    mShingle = pShingle;
    mShingleSize = pShingleSize;
    mBlockSize = pBlockSize;
    mHashAlgorithm = pHashAlgorithm;
}
InverseIndexCuda::~InverseIndexCuda() {
    cudaFree(mDev_FeatureList);
    cudaFree(mDev_ValuesList);
    cudaFree(mDev_SizeOfInstanceList);
    cudaFree(mDev_JumpLength);
    cudaFree(mDev_DotProduct);
}
void InverseIndexCuda::copyDataToGpu(SparseMatrixFloat* pRawData, int** pDevFeatureList,
                                      float** pDevValueList, int** pSizeList, int** pJumpList) {
    // size_t neededMemory = numberOfInstances / iterations  * signaturesSize * sizeof(int);
    // neededMemory += pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(int);
    // cudaMemGetInfo(&memory_free, &memory_total);
    // cudaFree(pDevFeatureList);
    // cudaFree(pDevValueList);
    // cudaFree(pSizeList);
    // cudaFree(pJumpList);
    
    int numberOfElements = 0;
    int* sizes = (int*) malloc(sizeof(int) * pRawData->size());
    int* jumpValues = (int*) malloc(sizeof(int) * pRawData->size());
    for (uint i = 0; i < pRawData->size(); ++i) {
        sizes[i] = static_cast<int>(pRawData->getInstance(i)->size());
        // printf("Sizes: %i\n", sizes[i]);
        jumpValues[i] = numberOfElements;
        numberOfElements += static_cast<int>(pRawData->getInstance(i)->size());
    }
    // memory for the number of features per instance
    cudaMalloc((void **) &(*pSizeList),
            static_cast<int>(pRawData->size()) * sizeof(int));
    // copy the size of all instances to the gpu               
    cudaMemcpy((*pSizeList), sizes,
            static_cast<int>(pRawData->size()) * sizeof(int),
            cudaMemcpyHostToDevice);
    free(sizes);
    printf("Inverse size pointer adress: %u\n", pSizeList);
    printf("Inverse &size pointer adress: %u\n", &pSizeList);
    
    // memory for the number of features per instance
    cudaMalloc((void **) &(*pJumpList),
            pRawData->size() * sizeof(int));
    // copy the size of all instances to the gpu               
    cudaMemcpy((*pJumpList), jumpValues,
            static_cast<int>(pRawData->size()) * sizeof(int),
            cudaMemcpyHostToDevice);
    // memory for instances and their featureIds
    cudaMalloc((void **) &(*pDevFeatureList),
            numberOfElements * sizeof(int));
    // memory for the values of the features of the instances
    cudaMalloc((void **) &(*pDevValueList), 
                numberOfElements * sizeof(float));
    
    
    int* dev_index = (int*) malloc(sizeof(int) * numberOfElements);
    float* dev_values = (float*) malloc(sizeof(float) * numberOfElements);
    
    for (uint i = 0; i < pRawData->size(); ++i) {
        std::vector<sparseData>* instance = pRawData->getInstance(i);
        for (uint j = 0; j < instance->size(); ++j) {
            dev_index[jumpValues[i] + j] = static_cast<int>((*instance)[j].instance);
            dev_values[jumpValues[i] + j] = static_cast<float>((*instance)[j].value);
            
        }
    }
    // copy instances and their feature ids to the gpu
    cudaMemcpy((*pDevFeatureList), dev_index,
                numberOfElements * sizeof(int),
            cudaMemcpyHostToDevice);
    
    cudaMemcpy((*pDevValueList), dev_values,
                numberOfElements * sizeof(float),
            cudaMemcpyHostToDevice);
    free(dev_index);
    free(dev_values);      
    free(jumpValues);
}
void InverseIndexCuda::computeSignaturesFittingOnGpu(SparseMatrixFloat* pRawData, 
                                                int pStartIndex, int pEndIndex, 
                                                int pNumberOfInstances, int pNumberOfBlocks, 
                                                int pNumberOfThreads, int pShingleFactor, 
                                                int pBlockSizeShingle,
                                                vvsize_t_p* pSignatures, int pRangeK) {
    // copy data to gpu
    
    copyDataToGpu(pRawData, &mDev_FeatureList, &mDev_ValuesList, &mDev_SizeOfInstanceList, &mDev_JumpLength);                                             
                                                    
    
    // check if enough memory is available on the gpu 
    size_t memory_total = 0;
    size_t memory_free = 0;
    int iterations = 1;
    int numberOfInstances = static_cast<int>(pEndIndex) - static_cast<int>(pStartIndex);
    int signaturesSize = mNumberOfHashFunctions * pBlockSizeShingle / pShingleFactor;
    
    // memory for all signatures and memory for signatures on each block
    size_t neededMemory = numberOfInstances / iterations  * signaturesSize * sizeof(int);
    neededMemory += pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(int);
    cudaMemGetInfo(&memory_free, &memory_total);
    // do i need more memory than it is free?
    if (neededMemory > memory_free) {
        iterations = ceil(neededMemory / static_cast<float>(memory_free));
    }
    
    int start = static_cast<int>(pStartIndex);
    int end = numberOfInstances / iterations;
  
    int windowSize = numberOfInstances / iterations;
    int* instancesHashValues = (int*) malloc(numberOfInstances / iterations * mNumberOfHashFunctions * sizeof(int));
    
    // memory for the inverse index on the gpu.
    // for each instance the number of hash functions
    cudaMalloc((void **) &mDev_ComputedSignaturesPerInstance,
            numberOfInstances / iterations  * signaturesSize * sizeof(int));
    int* dev_SignaturesBlockSize;
    cudaMalloc((void **) &dev_SignaturesBlockSize,
           pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(int));
     
     
    // cuda memory for dot products dot<X, X>
    cudaMalloc((void **) &mDev_DotProduct, sizeof(int) * numberOfInstances);
    // printf("start: %i, end: %i, iterations: %i\n", start, end, iterations);
    // compute the signatures on the gpu
    // do it in n iterations with equal sized chunks 
    // if the data would not fit on the ram of the gpu
    printf("%i\n", __LINE__);
     printf("inverse size pointer adress2: %u\n", mDev_SizeOfInstanceList);
    printf("inver &size pointer adress2: %u\n", &mDev_SizeOfInstanceList);
    for (uint i = 0; i < iterations; ++i) {
        // execute kernel on gpu
        if (mHashAlgorithm == 0) {
            fitCudaMinHash<<<pNumberOfBlocks, pNumberOfThreads>>>
            (mDev_FeatureList, 
            mDev_SizeOfInstanceList,  
            mNumberOfHashFunctions, 
            mDev_JumpLength,
                    mDev_ComputedSignaturesPerInstance, 
                    end, start, mBlockSize, mShingleSize, dev_SignaturesBlockSize);
        } else {
            fitCudaWtaHash<<<pNumberOfBlocks, pNumberOfThreads>>>
            (mDev_FeatureList, 
            mDev_SizeOfInstanceList,  
            mNumberOfHashFunctions, 
            mDev_JumpLength,
                    mDev_ComputedSignaturesPerInstance, 
                    end, start, mBlockSize, mShingleSize, dev_SignaturesBlockSize);
        }
    printf("%i\n", __LINE__);
    fflush(stdout);    
        dotProductSingle<<<32, 128>>>(mDev_FeatureList, mDev_ValuesList, mDev_SizeOfInstanceList,
                                        mDev_JumpLength, end, mDev_DotProduct);
    printf("%i\n", __LINE__);
                                        
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
    printf("%i\n", __LINE__);
        
        start = end+1;
        end = end + windowSize;
    }
    
    cudaFree(mDev_ComputedSignaturesPerInstance);
    cudaFree(dev_SignaturesBlockSize);
}
void InverseIndexCuda::computeSignaturesQueryOnGpu(SparseMatrixFloat* pRawData, 
                                                int pStartIndex, int pEndIndex, 
                                                int pNumberOfInstances, int pNumberOfBlocks, 
                                                int pNumberOfThreads, int pShingleFactor, 
                                                int pBlockSizeShingle,
                                                vvsize_t_p* pSignatures, int pRangeK) {
    int* dev_featureList;
    float* dev_valueList;
    int* dev_sizeOfInstanceList;
    int* jumpLengthList;
    int* dev_computedSignaturesPerInstance;
    int numberOfInstances = pEndIndex - pStartIndex;
    int signaturesSize = mNumberOfHashFunctions * pBlockSizeShingle / pShingleFactor;
    
    int memory_total = 0;
    int memory_free = 0;
    int iterations = 1;
    
    copyDataToGpu(pRawData, &dev_featureList, &dev_valueList, &dev_sizeOfInstanceList, &jumpLengthList);                                             
    
    
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
        if (mHashAlgorithm == 0) {
            fitCudaMinHash<<<pNumberOfBlocks, pNumberOfThreads>>>
            (dev_featureList, 
            dev_sizeOfInstanceList,  
            mNumberOfHashFunctions, 
            jumpLengthList,
                    dev_computedSignaturesPerInstance, 
                    end, start, mBlockSize, mShingleSize, dev_signaturesBlockSize);
        } else {
            fitCudaWtaHash<<<pNumberOfBlocks, pNumberOfThreads>>>
            (dev_featureList, 
            dev_sizeOfInstanceList,  
            mNumberOfHashFunctions, 
            jumpLengthList,
            dev_computedSignaturesPerInstance, 
            end, start, mBlockSize, mShingleSize, dev_signaturesBlockSize);
        }
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
    cudaFree(dev_sizeOfInstanceList);
    cudaFree(jumpLengthList);
}