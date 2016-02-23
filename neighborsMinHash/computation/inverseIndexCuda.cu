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
void InverseIndexCuda::computeSignaturesOnGpu(const SparseMatrixFloat* pRawData, 
                                                size_t pStartIndex, size_t pEndIndex, 
                                                size_t pNumberOfInstances, size_t pNumberOfBlocks, 
                                                size_t pNumberOfThreads, size_t pShingleFactor, 
                                                size_t pBlockSizeShingle,
                                                vvsize_t_p* pSignatures) {
    // check if enough memory is available on the gpu 
    size_t memory_total = 0;
    size_t memory_free = 0;
    size_t iterations = 1;
    cudaMemGetInfo(&memory_free, &memory_total);
    if (memory_free >= pRawData->getNumberOfInstances()  * mNumberOfHashFunctions * sizeof(size_t)) {
        iterations = ceil(pRawData->getNumberOfInstances()  * mNumberOfHashFunctions * sizeof(size_t) / static_cast<float>(memory_free));
    }
    
    size_t start = 0;
    size_t end = pRawData->getNumberOfInstances() / iterations;
    size_t windowSize = pRawData->getNumberOfInstances() / iterations;
    size_t* instancesHashValues = (size_t*) malloc(pRawData->getNumberOfInstances() / iterations * mNumberOfHashFunctions * sizeof(size_t));
    
    size_t signaturesSize = mNumberOfHashFunctions * pBlockSizeShingle / pShingleFactor;
    // memory for the inverse index on the gpu.
    // for each instance the number of hash functions
    cudaMalloc((void **) &mDev_ComputedSignaturesPerInstance,
            pRawData->getNumberOfInstances() / iterations  * signaturesSize * sizeof(size_t));
    size_t* dev_SignaturesBlockSize;
    cudaMalloc((void **) &dev_SignaturesBlockSize,
           pNumberOfBlocks * mNumberOfHashFunctions * pBlockSizeShingle * sizeof(size_t));
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
                    pRawData->getNumberOfInstances()/iterations * signaturesSize * sizeof(size_t),
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
    
    cudaFree(mDev_ComputedSignaturesPerInstance);
    cudaFree(dev_SignaturesBlockSize);
}

void InverseIndexCuda::computeHitsOnGpu(std::vector<vvsize_t_p*>* pHitsPerInstance, 
                                                neighborhood* pNeighborhood, 
                                                size_t pNeighborhoodSize,
                                                size_t pNumberOfInstances,
                                                size_t pNumberOfBlocks) {
    vsize_t hitsPerInstance;
    vsize_t sizePerInstance;
    size_t counter = 0;
    
    for (auto it = pHitsPerInstance->begin(); it != pHitsPerInstance->end(); ++it) {
        for (auto itQueryInstance = (*it)->begin(); itQueryInstance != (*it)->end(); ++itQueryInstance) {
            for(auto itInstance = (*itQueryInstance)->begin(); 
                itInstance != (*itQueryInstance)->end(); ++itInstance) {
                    hitsPerInstance.push_back(*itInstance);
                    ++counter;
            }
        }
        sizePerInstance.push_back(counter);
        counter = 0;
    }
    size_t* dev_HitsPerInstances;
    size_t* dev_SizePerInstances;
    int* dev_HistogramMemory;
    int* dev_SortingMemory;
    int* dev_RadixSortMemory;
    // std::cout << "Size of hitPerInstnace: " << hitsPerInstance.size() << std::endl;
    cudaMalloc((void **) &dev_HitsPerInstances,
            hitsPerInstance.size() * sizeof(size_t));
    cudaMalloc((void **) &dev_SizePerInstances,
            sizePerInstance.size() * sizeof(size_t));
    cudaMalloc((void **) &dev_HistogramMemory,
            pNumberOfBlocks*pNumberOfInstances * sizeof(int));
    cudaMalloc((void **) &dev_SortingMemory,
            pNumberOfBlocks * pNumberOfInstances * 2 * sizeof(int));
    cudaMalloc((void **) &dev_RadixSortMemory,
            pNumberOfBlocks * pNumberOfInstances * 2 * 2 * sizeof(int));
    cudaMemcpy(dev_HitsPerInstances, &hitsPerInstance[0],
                hitsPerInstance.size() * sizeof(size_t),
            cudaMemcpyHostToDevice);
    cudaMemcpy(dev_SizePerInstances, &sizePerInstance[0],
                sizePerInstance.size() * sizeof(size_t),
            cudaMemcpyHostToDevice);
    size_t* dev_Neighborhood;
    float* dev_Distances;
    int* neighborhood = (int*) malloc(pHitsPerInstance->size() * pNeighborhoodSize * sizeof(int));
    float* distances = (float*) malloc(pHitsPerInstance->size() * pNeighborhoodSize * sizeof(float));
    
    // size_t* instancesHashValues = (size_t*) malloc(pRawData->getNumberOfInstances() / iterations * mNumberOfHashFunctions * sizeof(size_t));
    
    cudaMalloc((void **) &dev_Neighborhood,
                pHitsPerInstance->size() * pNeighborhoodSize * sizeof(size_t));
    cudaMalloc((void **) &dev_Distances,
                pHitsPerInstance->size() * pNeighborhoodSize * sizeof(float));
    
    queryCuda<<<pNumberOfBlocks, pNumberOfInstances>>>
                (dev_HitsPerInstances, dev_SizePerInstances,
                pNeighborhoodSize, dev_Neighborhood,
                dev_Distances, pHitsPerInstance->size(), dev_HistogramMemory,
                dev_RadixSortMemory, dev_SortingMemory);
    
    if (!pFast) {
        if (pDistance) {
            euclideanDistanceCuda<<<512, 32>>>(dev_SortingMemory,
            );
        } else {
            cosineSimilarityCuda<<<512, 32>>>(dev_SortingMemory,
            );
        }
    }
    
    cudaMemcpy(neighborhood, dev_Neighborhood,
                pHitsPerInstance->size() * pNeighborhoodSize * sizeof(int),
                cudaMemcpyDeviceToHost);
    cudaMemcpy(distances, dev_Distances,
                pHitsPerInstance->size() * pNeighborhoodSize * sizeof(float),
                cudaMemcpyDeviceToHost);
                
     cudaFree(dev_Neighborhood);
     cudaFree(dev_Distances);
     // transfer to neighorhood layout
     vvint* neighborsVector = new vvint(pHitsPerInstance->size());
     vvfloat* distancesVector = new vvfloat(pHitsPerInstance->size());
     
     for (size_t i = 0; i < pHitsPerInstance->size(); ++i) {
         vint neighbors_;
         vfloat distances_;
         for (size_t j = 0; j < pNeighborhoodSize; ++j) {
             neighbors_.push_back(neighborhood[i*pNeighborhoodSize + j]);
             distances_.push_back(distances[i*pNeighborhoodSize + j]);
         }
         (*neighborsVector)[i] = neighbors_;
         (*distancesVector)[i] = distances_;
     }
     
     pNeighborhood->neighbors = neighborsVector;
     pNeighborhood->distances = distancesVector;
     free(neighborhood);
     free(distances);
     
     // return it
     // delete memory
     // 
     // check if everything is fitting in gpu memory,
     // loop if not.              
}
