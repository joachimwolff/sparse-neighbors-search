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
    // printf("mDev_FeatureList, %u\n", mDev_FeatureList);
    // printf("&mDev_FeatureList, %u\n", &mDev_FeatureList);
    // printf("mDev_ValuesList, %u\n", mDev_ValuesList);
    // printf("&mDev_ValuesList, %u\n", &mDev_ValuesList);
    // printf("mDev_SizeOfInstanceList, %u\n", mDev_SizeOfInstanceList);
    // printf("&mDev_SizeOfInstanceList, %u\n", &mDev_SizeOfInstanceList);
    // printf("%i\n", __LINE__);
    copyDataToGpu(pRawData, &mDev_FeatureList, &mDev_ValuesList, &mDev_SizeOfInstanceList);  
    // printf("%i\n", __LINE__);
                                               
    size_t signaturesSize = ceil(mNumberOfHashFunctions * pBlockSizeShingle / (float) pShingleFactor);
    // printf("%i\n", __LINE__);
   
    int* instancesHashValues = (int*) malloc(pRawData->size() * signaturesSize * sizeof(int));
    // printf("%i\n", __LINE__);
    
    // memory for the inverse index on the gpu.
    // for each instance the number of hash functions
    cudaMalloc((void **) &mDev_ComputedSignaturesPerInstance,
            pRawData->size() * signaturesSize * sizeof(int));
    int* dev_SignaturesBlockSize;
    cudaMalloc((void **) &dev_SignaturesBlockSize,
           128 * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(int));
    // printf("%i\n", __LINE__);
     
     
    // cuda memory for dot products dot<X, X>
    cudaMalloc((void **) &mDev_DotProduct, sizeof(float) * pRawData->size());
  
    // printf("%i\n", __LINE__);
 
    
        // execute kernel on gpu
        if (mHashAlgorithm == 0) {
    // printf("%i\n", __LINE__);
            
            fitCudaMinHash<<<128, 128>>>
            (mDev_FeatureList, 
            mDev_SizeOfInstanceList,  
            mNumberOfHashFunctions, 
            pRawData->getMaxNnz(),
            mDev_ComputedSignaturesPerInstance, 
            pRawData->size(), 0, mBlockSize, mShingleSize, dev_SignaturesBlockSize);
            cudaDeviceSynchronize();
    // printf("%i\n", __LINE__);
            
        } else {
            // fitCudaWtaHash<<<128, 128>>>
            // (mDev_FeatureList, 
            // mDev_SizeOfInstanceList,  
            // mNumberOfHashFunctions, 
            // mDev_JumpLength,
            //         mDev_ComputedSignaturesPerInstance, 
            //         end, start, mBlockSize, mShingleSize, dev_SignaturesBlockSize);
        }
        // dotProductSingle<<<128, 128>>>(mDev_FeatureList, mDev_ValuesList, mDev_SizeOfInstanceList,
        //                                 mDev_JumpLength, pRawData->size(), mDev_DotProduct);
        //     cudaDeviceSynchronize();
    // printf("%i\n", __LINE__);
                                        
        // copy results back to host  
        cudaMemcpy(instancesHashValues, mDev_ComputedSignaturesPerInstance, 
                    pRawData->size() * signaturesSize * sizeof(int),
                    cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    // printf("%i\n", __LINE__);
                    
        // copy values into one vector per instance
        for(size_t i = 0; i < pRawData->size(); ++i) {
            vsize_t* instance = new vsize_t(signaturesSize);
            for (size_t j = 0; j < signaturesSize; ++j) {
                (*instance)[j] = static_cast<size_t> (instancesHashValues[i*signaturesSize + j]);
            }
            (*pSignatures)[i] = instance;
        }
    // printf("%i\n", __LINE__);
    
    cudaFree(mDev_ComputedSignaturesPerInstance);
    cudaFree(dev_SignaturesBlockSize);
    dotProductSingle<<<128, 128>>>(mDev_FeatureList, mDev_ValuesList, mDev_SizeOfInstanceList,
                                        pRawData->size(), pRawData->getMaxNnz(), mDev_DotProduct);
    cudaDeviceSynchronize();
    printf("DotproductInverse: %u\n", mDev_DotProduct);
}
void InverseIndexCuda::computeSignaturesQueryOnGpu(SparseMatrixFloat* pRawData, 
                                                size_t pStartIndex, size_t pEndIndex, 
                                                size_t pNumberOfInstances, size_t pNumberOfBlocks, 
                                                size_t pNumberOfThreads, size_t pShingleFactor, 
                                                size_t pBlockSizeShingle,
                                                vvsize_t_p* pSignatures, size_t pRangeK) {
                                                    
//    // memory for the number of features per instance
//     cudaMalloc((void **) &(*pSizeList),
//            sizeof(size_t) * pRawData->size());
//     // copy the size of all instances to the gpu               
//     cudaMemcpy((*pSizeList), pRawData->getSparseMatrixSizeOfInstances(),
//             sizeof(size_t) * pRawData->size(),
//             cudaMemcpyHostToDevice);
    
//     // memory for instances and their featureIds
//     cudaMalloc((void **) &(*pDevFeatureList),
//             pRawData->size() * pRawData->getMaxNnz() * sizeof(int));
//     // memory for the values of the features of the instances
//     cudaMalloc((void **) &(*pDevValueList), 
//                 pRawData->size() * pRawData->getMaxNnz() * sizeof(float));
    
//     // copy instances and their feature ids to the gpu
//     cudaMemcpy((*pDevFeatureList), pRawData->getSparseMatrixIndex(),
//                 pRawData->size() * pRawData->getMaxNnz() * sizeof(int),
//             cudaMemcpyHostToDevice);
    
//     cudaMemcpy((*pDevValueList), pRawData->getSparseMatrixValues(),
//                 pRawData->size() * pRawData->getMaxNnz() * sizeof(float),
//             cudaMemcpyHostToDevice);
                                                           
//    size_t signaturesSize = ceil(mNumberOfHashFunctions * pBlockSizeShingle / (float) pShingleFactor);
   
//     size_t* instancesHashValues = (size_t*) malloc(pRawData->size() * signaturesSize * sizeof(size_t));
    
//     // memory for the inverse index on the gpu.
//     // for each instance the number of hash functions
//     cudaMalloc((void **) &mDev_ComputedSignaturesPerInstance,
//             pRawData->size() * signaturesSize * sizeof(int));
//     int* dev_SignaturesBlockSize;
//     cudaMalloc((void **) &dev_SignaturesBlockSize,
//            128 * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(int));
     
     
//     // cuda memory for dot products dot<X, X>
//     cudaMalloc((void **) &mDev_DotProduct, sizeof(float) * pRawData->size());
  
 
    
//         // execute kernel on gpu
//         if (mHashAlgorithm == 0) {
//             fitCudaMinHash<<<128, 128>>>
//             (mDev_FeatureList, 
//             mDev_SizeOfInstanceList,  
//             mNumberOfHashFunctions, 
//             pRawData->getMaxNnz(),
//             mDev_ComputedSignaturesPerInstance, 
//             pRawData->size(), 0, mBlockSize, mShingleSize, dev_SignaturesBlockSize);
//             cudaDeviceSynchronize();
//         } else {
//             // fitCudaWtaHash<<<128, 128>>>
//             // (mDev_FeatureList, 
//             // mDev_SizeOfInstanceList,  
//             // mNumberOfHashFunctions, 
//             // mDev_JumpLength,
//             //         mDev_ComputedSignaturesPerInstance, 
//             //         end, start, mBlockSize, mShingleSize, dev_SignaturesBlockSize);
//         }
//         // dotProductSingle<<<128, 128>>>(mDev_FeatureList, mDev_ValuesList, mDev_SizeOfInstanceList,
//         //                                 mDev_JumpLength, pRawData->size(), mDev_DotProduct);
//         //     cudaDeviceSynchronize();
                                        
//         // copy results back to host  
//         cudaMemcpy(instancesHashValues, mDev_ComputedSignaturesPerInstance, 
//                     pRawData->size() * signaturesSize * sizeof(int),
//                     cudaMemcpyDeviceToHost);
//         cudaDeviceSynchronize();
                    
//         // copy values into one vector per instance
//         for(size_t i = 0; i < pRawData->size(); ++i) {
//             vsize_t* instance = new vsize_t(signaturesSize);
//             for (size_t j = 0; j < signaturesSize; ++j) {
//                 (*instance)[j] = static_cast<size_t> (instancesHashValues[i*signaturesSize + j]);
//             }
//             (*pSignatures)[i] = instance;
//         }
    
//     cudaFree(mDev_ComputedSignaturesPerInstance);
//     cudaFree(dev_SignaturesBlockSize);
//     // dotProductSingle<<<128, 128>>>(mDev_FeatureList, mDev_ValuesList, mDev_SizeOfInstanceList,
//     //                                     pRawData->size(), pRawData->getMaxNnz(), mDev_DotProduct);
//     cudaDeviceSynchronize();
}