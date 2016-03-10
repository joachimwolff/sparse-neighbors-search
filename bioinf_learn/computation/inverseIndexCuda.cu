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
void InverseIndexCuda::computeHitsOnGpu(std::vector<vvsize_t_p*>* pHitsPerInstance, 
                                                neighborhood* pNeighborhood, 
                                                size_t pNeighborhoodSize,
                                                size_t pNumberOfInstances,
                                                const size_t pNumberOfBlocksHistogram,
                                                const size_t pNumberOfThreadsHistogram,
                                                const size_t pNumberOfBlocksDistance,
                                                const size_t pNumberOfThreadsDistance,
                                                size_t pFast, size_t pDistance,
                                                size_t pExcessFactor, size_t pMaxNnz) {
                                                    
//     struct hits {
//     size_t* instances;
//     size_t size;
// };
    // size_t* hitsPerInstance_realloc = NULL;
    //  = (size_t*) malloc(pHitsPerInstance->)
    size_t* elementsPerInstance = (size_t*) malloc(pHitsPerInstance->size() * sizeof(size_t));
    size_t counter = 0;
    
    hits* data = (hits*) malloc(sizeof(hits) * pHitsPerInstance->size());
    hits* dev_data_inner = (hits*) malloc(sizeof(hits) * pHitsPerInstance->size());
    // size_t i = 0;
    // return;
    // std::cout << "Size of pHits: " << pHitsPerInstance->size() << " pNumberofInstances: " << pNumberOfInstances;
    // std::cout << std::endl;
    size_t counterHitsPerInstance = 0;
    for (size_t i = 0; i < pHitsPerInstance->size(); ++i) {
        for (auto itQueryInstance = (*pHitsPerInstance)[i]->begin(); 
                itQueryInstance != (*pHitsPerInstance)[i]->end(); ++itQueryInstance) {
            for(auto itInstance = (*itQueryInstance)->begin(); 
                itInstance != (*itQueryInstance)->end(); ++itInstance) {
                    // hitsPerInstance[counterHitsPerInstance] = *itInstance;
                    ++counter;
            }
        }
        size_t* instances = (size_t*) malloc(sizeof(size_t) * counter);
        size_t j = 0;
        for (auto itQueryInstance = (*pHitsPerInstance)[i]->begin(); 
                itQueryInstance != (*pHitsPerInstance)[i]->end(); ++itQueryInstance) {
            for(auto itInstance = (*itQueryInstance)->begin(); 
                itInstance != (*itQueryInstance)->end(); ++itInstance) {
                    instances[j] = *itInstance;
                    ++j;
            }
        }
        data[i].instances = instances;
        data[i].size = counter;
        counter = 0;
    }
    memcpy(dev_data_inner, data, sizeof(hits) * pHitsPerInstance->size()));
    
    for (size_t i = 0; i < pHitsPerInstance->size(); ++i) {
        cudaMalloc(&(dev_data[i].instances), data[i].size*sizeof(size_t));
        cudaMemcpy(dev_data[i].instances, data[i].instances, data[i].size*sizeof(size_t));
    }
    hits* dev_data;
    cudaMalloc(&dev_data, sizeof(hits) * pHitsPerInstance->size());
    cudaMemcpy(dev_data, dev_data_inner, sizeof(hits) * pHitsPerInstance->size());
    
    // histogram* pHistogram, radixSortingMemory* pRadixSortMemory,
                                            // sortedHistogram* pHistogramSorted, 
    // reserve space for histogram on device
    histogram* dev_histogram_inner = (histogram*) malloc(pNumberOfBlocksHistogram*sizeof(histogram));
    for (size_t i = 0; i < pNumberOfBlocksHistogram; ++i) {
        cudaMalloc(&(dev_histogram_inner[i].instances), pNumberOfInstances * sizeof(size_t));
    }
    histogram* dev_histogram;
    cudaMalloc(&dev_histogram, pNumberOfBlocksHistogram*sizeof(histogram));
    cudaMemcpy(dev_histogram, dev_histogram_inner, pNumberOfBlocksHistogram*sizeof(histogram));
    
    
    // reserve for sorting on device
    radixSortingMemory* dev_radixSortingMemory_inner = (radixSortingMemory*) malloc(pNumberOfBlocksHistogram*sizeof(radixSortingMemory));
    for (size_t i = 0; i < pNumberOfBlocksHistogram; ++i) {
        cudaMalloc(&(dev_radixSortingMemory_inner[i].bucketNull), pNumberOfInstances * sizeof(int2));
        cudaMalloc(&(dev_radixSortingMemory_inner[i].bucketOne), pNumberOfInstances * sizeof(int2));
    }
    
    radixSortingMemory* dev_radixSortingMemory;
    cudaMalloc(&dev_radixSortingMemory, pNumberOfBlocksHistogram*sizeof(radixSortingMemory));
    cudaMemcpy(dev_radixSortingMemory, dev_radixSortingMemory_inner, pNumberOfBlocksHistogram*sizeof(radixSortingMemory));
    
    // space for sorted histogram
    sortedHistogram* dev_sortedHistogram_inner = (sortedHistogram*) malloc(pNumberOfBlocksHistogram*sizeof(sortedHistogram));
    for (size_t i = 0; i < pNumberOfBlocksHistogram; ++i) {
        cudaMalloc(&(dev_sortedHistogram_inner[i].instances), pNumberOfInstances * sizeof(int2));
    }
    
    sortedHistogram* dev_sortedHistogram;
    cudaMalloc(&dev_sortedHistogram, pNumberOfBlocksHistogram*sizeof(sortedHistogram));
    cudaMemcpy(dev_sortedHistogram, dev_sortedHistogram_inner, pNumberOfBlocksHistogram*sizeof(sortedHistogram));
    
    
    // printf("foo"); 
    size_t memory_free;
    size_t memory_total;
    size_t iterations = 1;
    vsize_t instancesPerIteration = {pNumberOfInstances};
    // size_t 
    // memory for all instances and their hits
    size_t needed_memory = counterHitsPerInstance * sizeof(size_t);
    // memory for the histogram
    needed_memory += pNumberOfBlocksHistogram*pNumberOfInstances * sizeof(int);
    // memory for number of elements per instance
    needed_memory += pNumberOfInstances * sizeof(size_t);
    // memory for radix sort
    needed_memory += pNumberOfBlocksHistogram * pNumberOfInstances * 2 * 2 * sizeof(int);
    // memory for sorted instances
    needed_memory += pNumberOfBlocksHistogram * pNumberOfInstances * 2 * sizeof(int);
    // get memory usage from gpu
    cudaMemGetInfo(&memory_free, &memory_total);
    // enough memory on the gpu plus an buffer of 1MB
    // if (memory_free >= needed_memory+1024*8*1024) {
    //     iterations = ceil(needed_memory / static_cast<float>(memory_free));
    //     if (iterations > elementsPerInstance.size()) {
    //         printf("Sorry your dataset is too big to be computed on your GPU. Please use CPU only mode");
    //     }
    //     size_t elementsPerIteration = elementsPerInstance.size() / iteration;
    //     for (size_t i = 0; i < iterations; ++i) {
    //         size_t numberOfElements = 0;
    //         for (size_t j = elementsPerIteration * i; j < elementsPerIteration ; ++j) {
    //               numberOfElements += elementsPerIteration[j];
    //         }
    //         instancesPerIteration.push_back(numberOfElements);
    //     }
    // }
    
    size_t* dev_Neighborhood;
    float* dev_Distances;
    size_t* neighborhood = (size_t*) malloc(pHitsPerInstance->size() * pNeighborhoodSize * sizeof(size_t));
    float* distances = (float*) malloc(pHitsPerInstance->size() * pNeighborhoodSize * sizeof(float));
    int2* histogram = (int2*) malloc(pNumberOfBlocksHistogram * pNumberOfInstances * sizeof(int2));
    // size_t* instancesHashValues = (size_t*) malloc(pRawData->getNumberOfInstances() / iterations * mNumberOfHashFunctions * sizeof(size_t));
    
    cudaMalloc((void **) &dev_Neighborhood,
                pHitsPerInstance->size() * pNeighborhoodSize * pExcessFactor * sizeof(size_t));
    cudaMalloc((void **) &dev_Distances,
                pHitsPerInstance->size() * pNeighborhoodSize * sizeof(float));
    vvint* neighborsVector = new vvint(pHitsPerInstance->size());
    vvfloat* distancesVector = new vvfloat(pHitsPerInstance->size());
    for (size_t i = 0; i < iterations; ++i) {
        createSortedHistogramsCuda<<<pNumberOfBlocksHistogram, pNumberOfThreadsHistogram>>>
                    (dev_data, pNumberOfInstances, dev_histogram,
                    dev_radixSortingMemory, dev_sortedHistogram, 
                    dev_NumberOfPossibleNeighbors, 
                    pNeighborhoodSize, pExcessFactor,
                    dev_Neighborhood, dev_Distances, pFast);
        cudaMemcpy(histogram, dev_HistogramSorted, pNumberOfBlocksHistogram * pNumberOfInstances *sizeof(int2)
                        ,cudaMemcpyDeviceToHost); 
        printf("pNumberOfInstances: %i", pNumberOfInstances);
        printf("\n\nunsorted histogram: ");
        for (size_t j = 0; j <  pNumberOfInstances; j += 1) {
                printf("id: %i, count: %i", histogram[j].y, histogram[j].x);
        }
        printf("\n\n");
        if (!pFast) {
            if (pDistance) {
                euclideanDistanceCuda<<<pNumberOfBlocksDistance, pNumberOfThreadsDistance>>>
                                        (dev_sortedHistogram, dev_NumberOfPossibleNeighbors, 
                                        pNumberOfInstances,
                                        mDev_FeatureList, mDev_ValuesList,
                                        mDev_SizeOfInstanceList, pMaxNnz,
                                        dev_RadixSortMemory, pNeighborhoodSize,
                                        dev_Neighborhood, dev_Distances);
                
            } else {
                cosineSimilarityCuda<<<pNumberOfBlocksDistance, pNumberOfThreadsDistance>>>
                                        (dev_HistogramSorted, dev_NumberOfPossibleNeighbors, 
                                        rangeBetweenInstances, pNumberOfInstances,
                                        mDev_FeatureList, mDev_ValuesList,
                                        mDev_SizeOfInstanceList, pMaxNnz,
                                        dev_RadixSortMemory, pNeighborhoodSize,
                                        dev_Neighborhood, dev_Distances);
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
    }
     
    pNeighborhood->neighbors = neighborsVector;
    pNeighborhood->distances = distancesVector;
    free(neighborhood);
    free(distances);
    free(elementsPerInstance);
    cudaFree(dev_HitsPerInstances);
    cudaFree(dev_ElementsPerInstances);
    cudaFree(dev_Histogram);
    cudaFree(dev_HistogramSorted);
    cudaFree(dev_RadixSortMemory);
    cudaFree(dev_NumberOfPossibleNeighbors);
     // return it
     // delete memory
     // 
     // check if everything is fitting in gpu memory,
     // loop if not.              
}
