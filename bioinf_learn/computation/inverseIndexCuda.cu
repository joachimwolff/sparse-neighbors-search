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
    cudaFree(mDev_FeatureList);
    cudaFree(mDev_ValuesList);
    cudaFree(mDev_SizeOfInstanceList);
    cudaFree(mDev_JumpLength);
    cudaFree(mDev_DotProduct);
}
void InverseIndexCuda::copyDataToGpu(SparseMatrixFloat* pRawData, size_t** pDevFeatureList,
                                      size_t** pDevValueList, size_t** pSizeList) {

    // memory for the number of features per instance
    cudaMalloc((void **) &(*pSizeList),
           sizeof(size_t) * pRawData->size());
    // copy the size of all instances to the gpu               
    cudaMemcpy((*pSizeList), pRawData->getSparseMatrixSizeOfInstances(),
            sizeof(size_t) * pRawData->size(),
            cudaMemcpyHostToDevice);
    
    // memory for instances and their featureIds
    cudaMalloc((void **) &(*pDevFeatureList),
            pRawData->size() * pRawData->getMaxNnz() * sizeof(size_t));
    // memory for the values of the features of the instances
    cudaMalloc((void **) &(*pDevValueList), 
                pRawData->size() * pRawData->getMaxNnz() * sizeof(size_t);
    
    // copy instances and their feature ids to the gpu
    cudaMemcpy((*pDevFeatureList), pRawData->getSparseMatrixIndex(),
                pRawData->size() * pRawData->getMaxNnz() * sizeof(size_t),
            cudaMemcpyHostToDevice);
    
    cudaMemcpy((*pDevValueList), pRawData->getSparseMatrixValues(),
                pRawData->size() * pRawData->getMaxNnz() * sizeof(size_t),,
            cudaMemcpyHostToDevice);
 
    // printf("Feature list pointer adress: %u\n", (*pDevFeatureList));
    // printf("&Feature list pointer adress: %u\n", &(*pDevFeatureList));
    // printf("pDevValueList pointer adress: %u\n", (*pDevValueList));
    // printf("&pDevValueList pointer adress: %u\n", &(*pDevValueList));
    // printf("Inverse size pointer adress: %u\n", (*pSizeList));
    // printf("Inverse &size pointer adress: %u\n", &(*pSizeList));
    // printf("jumppointer adress: %u\n", (*pJumpList));
    // printf("&jump pointer adress: %u\n", &(*pJumpList));
}
void InverseIndexCuda::computeSignaturesFittingOnGpu(SparseMatrixFloat* pRawData, 
                                                size_t pStartIndex, size_t pEndIndex, 
                                                size_t pNumberOfInstances, size_t pNumberOfBlocks, 
                                                size_t pNumberOfThreads, size_t pShingleFactor, 
                                                size_t pBlockSizeShingle,
                                                vvsize_t_p* pSignatures, size_t pRangeK) {
    // copy data to gpu
    
    copyDataToGpu(pRawData, &mDev_FeatureList, &mDev_ValuesList, &mDev_SizeOfInstanceList);                                             
                                                    
    
    // check if enough memory is available on the gpu 
    // size_t memory_total = 0;
    // size_t memory_free = 0;
    // int iterations = 1;
    // int numberOfInstances = pEndIndex) - static_cast<int>(pStartIndex);
    size_t signaturesSize = ceil(mNumberOfHashFunctions * pBlockSizeShingle / (float) pShingleFactor);
    
    // // memory for all signatures and memory for signatures on each block
    // size_t neededMemory = numberOfInstances / iterations  * signaturesSize * sizeof(int);
    // neededMemory += pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(int);
    // cudaMemGetInfo(&memory_free, &memory_total);
    // // do i need more memory than it is free?
    // if (neededMemory > memory_free) {
    //     iterations = ceil(neededMemory / static_cast<float>(memory_free));
    // }
    
    // int start = static_cast<int>(pStartIndex);
    // int end = numberOfInstances / iterations;
  
    // int windowSize = numberOfInstances / iterations;
    // int* instancesHashValues = (int*) malloc(numberOfInstances / iterations * mNumberOfHashFunctions * sizeof(int));
    
    // memory for the inverse index on the gpu.
    // for each instance the number of hash functions
    cudaMalloc((void **) &mDev_ComputedSignaturesPerInstance,
            pRawData->size() * signaturesSize * sizeof(size_t));
    size_t* dev_SignaturesBlockSize;
    cudaMalloc((void **) &dev_SignaturesBlockSize,
           128 * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(size_t));
     
     
    // cuda memory for dot products dot<X, X>
    cudaMalloc((void **) &mDev_DotProduct, sizeof(int) * numberOfInstances);
    // printf("start: %i, end: %i, iterations: %i\n", start, end, iterations);
    // compute the signatures on the gpu
    // do it in n iterations with equal sized chunks 
    // if the data would not fit on the ram of the gpu
    printf("%i\n", __LINE__);
    printf("inverse size pointer adress2: %u\n", mDev_SizeOfInstanceList);
    printf("inver &size pointer adress2: %u\n", &mDev_SizeOfInstanceList);
    // for (size_t i = 0; i < iterations; ++i) {
        // execute kernel on gpu
        if (mHashAlgorithm == 0) {
            fitCudaMinHash<<<128, 128>>>
            (mDev_FeatureList, 
            mDev_SizeOfInstanceList,  
            mNumberOfHashFunctions, 
            pRawData->getMaxNnz(),
                    mDev_ComputedSignaturesPerInstance, 
                    end, start, mBlockSize, mShingleSize, dev_SignaturesBlockSize);
        } else {
            // fitCudaWtaHash<<<128, 128>>>
            // (mDev_FeatureList, 
            // mDev_SizeOfInstanceList,  
            // mNumberOfHashFunctions, 
            // mDev_JumpLength,
            //         mDev_ComputedSignaturesPerInstance, 
            //         end, start, mBlockSize, mShingleSize, dev_SignaturesBlockSize);
        }
        dotProductSingle<<<32, 128>>>(mDev_FeatureList, mDev_ValuesList, mDev_SizeOfInstanceList,
                                        mDev_JumpLength, end, mDev_DotProduct);
                                        
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
    // printf("%i\n", __LINE__);
        
        // start = end+1;
        // end = end + windowSize;
    // }
    
    cudaFree(mDev_ComputedSignaturesPerInstance);
    cudaFree(dev_SignaturesBlockSize);
}
void InverseIndexCuda::computeSignaturesQueryOnGpu(SparseMatrixFloat* pRawData, 
                                                size_t pStartIndex, size_t pEndIndex, 
                                                size_t pNumberOfInstances, size_t pNumberOfBlocks, 
                                                size_t pNumberOfThreads, size_t pShingleFactor, 
                                                size_t pBlockSizeShingle,
                                                vvsize_t_p* pSignatures, size_t pRangeK) {
    // size_t* dev_featureList;
    // size_t* dev_valueList;
    // size_t* dev_sizeOfInstanceList;
    // // int* jumpLengthList;
    // size_t* dev_computedSignaturesPerInstance;
    // size_t numberOfInstances = pEndIndex - pStartIndex;
    // size_t signaturesSize = mNumberOfHashFunctions * pBlockSizeShingle / pShingleFactor;
    
    // size_t memory_total = 0;
    // size_t memory_free = 0;
    // size_t iterations = 1;
    
    // copyDataToGpu(pRawData, &dev_featureList, &dev_valueList, &dev_sizeOfInstanceList);                                             
    
    
    // size_t start = 0;
    // size_t end = numberOfInstances / iterations;
    // size_t windowSize = numberOfInstances / iterations;
    // size_t* instancesHashValues = (int*) malloc(numberOfInstances / iterations * mNumberOfHashFunctions * sizeof(int));
    
    // // size_t signaturesSize = mNumberOfHashFunctions * pBlockSizeShingle / pShingleFactor;
    // // memory for the signatures on the gpu.
    // // for each instance the number of hash functions
    // cudaMalloc((void **) &dev_computedSignaturesPerInstance,
    //         numberOfInstances / iterations  * signaturesSize * sizeof(int));
    // int* dev_signaturesBlockSize;
    // cudaMalloc((void **) &dev_signaturesBlockSize,
    //        pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(int));
    // // compute the signatures on the gpu
    // // do it in n iterations with equal sized chunks 
    // // if the data would not fit on the ram of the gpu
    // for (size_t i = 0; i < iterations; ++i) {
    //     // execute kernel on gpu
    //     if (mHashAlgorithm == 0) {
    //         fitCudaMinHash<<<pNumberOfBlocks, pNumberOfThreads>>>
    //         (dev_featureList, 
    //         dev_sizeOfInstanceList,  
    //         mNumberOfHashFunctions, 
    //         jumpLengthList,
    //                 dev_computedSignaturesPerInstance, 
    //                 end, start, mBlockSize, mShingleSize, dev_signaturesBlockSize);
    //     } else {
    //         fitCudaWtaHash<<<pNumberOfBlocks, pNumberOfThreads>>>
    //         (dev_featureList, 
    //         dev_sizeOfInstanceList,  
    //         mNumberOfHashFunctions, 
    //         jumpLengthList,
    //         dev_computedSignaturesPerInstance, 
    //         end, start, mBlockSize, mShingleSize, dev_signaturesBlockSize);
    //     }
    //     // copy results back to host      
    //     cudaMemcpy(instancesHashValues, dev_computedSignaturesPerInstance, 
    //                 numberOfInstances/iterations * signaturesSize * sizeof(int),
    //                 cudaMemcpyDeviceToHost);
    //     // copy values into one vector per instance
    //     for(size_t i = start; i < end; ++i) {
    //         vsize_t* instance = new vsize_t(signaturesSize);
    //         for (size_t j = 0; j < signaturesSize; ++j) {
    //             (*instance)[j] = static_cast<size_t>(instancesHashValues[i*signaturesSize + j]);
    //         }
    //         (*pSignatures)[i] = instance;
    //     }
        
    //     start = end+1;
    //     end = end + windowSize;
    // }
    
    // cudaFree(dev_computedSignaturesPerInstance);
    // cudaFree(dev_signaturesBlockSize);
    // cudaFree(dev_featureList);       
    // cudaFree(dev_computedSignaturesPerInstance);       
    // cudaFree(dev_sizeOfInstanceList);
    // cudaFree(jumpLengthList);
}