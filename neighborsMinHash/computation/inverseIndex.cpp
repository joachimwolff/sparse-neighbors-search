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

#include <algorithm>
#include <iostream>
#include <iterator>
#include <utility>

#ifdef OPENMP
#include <omp.h>
#endif

#include <time.h>
#include "inverseIndex.h"
#include "kSizeSortedMap.h"
// #include "inverseIndexCuda.h"

class sort_map {
  public:
    size_t key;
    size_t val;
};

bool mapSortDescByValue(const sort_map& a, const sort_map& b) {
        return a.val > b.val;
};
InverseIndex::InverseIndex(){};
InverseIndex::InverseIndex(size_t pNumberOfHashFunctions, size_t pShingleSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions, size_t pBloomierFilter,
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance, int pRemoveHashFunctionWithLessEntriesAs,
                    size_t pHashAlgorithm, size_t pBlockSize, size_t pShingle, size_t pRemoveValueWithLeastSigificantBit) {   
    mNumberOfHashFunctions = pNumberOfHashFunctions;
    mShingleSize = pShingleSize;
    mNumberOfCores = pNumberOfCores;
    mChunkSize = pChunkSize;
    mMaxBinSize = pMaxBinSize;
    mMinimalBlocksInCommon = pMinimalBlocksInCommon;
    mExcessFactor = pExcessFactor;
    mMaximalNumberOfHashCollisions = pMaximalNumberOfHashCollisions;
    mPruneInverseIndex = pPruneInverseIndex;
    mPruneInverseIndexAfterInstance = pPruneInverseIndexAfterInstance;
    mRemoveHashFunctionWithLessEntriesAs = pRemoveHashFunctionWithLessEntriesAs;
    mHashAlgorithm = pHashAlgorithm;
    mSignatureStorage = new umap_uniqueElement();
    mHash = new Hash();
    mBlockSize = pBlockSize;
    mShingle = pShingle;
    
    if (mShingle == 0) {
        if (mBlockSize == 0) {
            mBlockSize = 1;
        }
        mInverseIndexSize = mNumberOfHashFunctions * mBlockSize;
        mShingleSize = 1;
        mBlockSize = 1;
    } else {
        mInverseIndexSize = ceil(((float) (mNumberOfHashFunctions * mBlockSize) / (float) mShingleSize));
        std::cout << "size inverse index: " << mInverseIndexSize << std::endl;      
    }
        mInverseIndexStorage = new InverseIndexStorageUnorderedMap(mInverseIndexSize, mMaxBinSize);
    mRemoveValueWithLeastSigificantBit = pRemoveValueWithLeastSigificantBit;
    #ifdef CUDA
    mInverseIndexCuda = new InverseIndexCuda(pNumberOfHashFunctions, mShingle, mShingleSize, mBlockSize);
    #endif
}
 
InverseIndex::~InverseIndex() {
    for (auto it = mSignatureStorage->begin(); it != mSignatureStorage->end(); ++it) {
        // if ((*it).second.instances != NULL) {
            delete (*it).second.instances;
        // }
        // if ((*it).second.signature != NULL) {
            delete (*it).second.signature;
        // }
    }
    delete mSignatureStorage;
    delete mHash;
    delete mInverseIndexStorage;
}

distributionInverseIndex* InverseIndex::getDistribution() {
    return mInverseIndexStorage->getDistribution();
}

 // compute the signature for one instance
vsize_t* InverseIndex::computeSignature(const SparseMatrixFloat* pRawData, const size_t pInstance) {
    vsize_t* signature = new vsize_t(mNumberOfHashFunctions * mBlockSize);

    for(size_t j = 0; j < mNumberOfHashFunctions * mBlockSize; ++j) {
            size_t minHashValue = MAX_VALUE;        
            for (size_t i = 0; i < pRawData->getSizeOfInstance(pInstance); ++i) {
                size_t hashValue = mHash->hash((pRawData->getNextElement(pInstance, i) +1), (j+1), MAX_VALUE);
                if (hashValue < minHashValue) {
                    minHashValue = hashValue;
                }
            }
            (*signature)[j] = minHashValue;
    }
    // reduce number of hash values by a factor of mShingleSize
    if (mShingle) {
        return shingle(signature);
    }
    return signature;
}

vsize_t* InverseIndex::shingle(vsize_t* pSignature) {
    
    vsize_t* signature = new vsize_t(mInverseIndexSize);
    size_t iterationSize = (mNumberOfHashFunctions * mBlockSize) / mShingleSize;
    if (mShingle == 1) {
        
        // if 0 than combine hash values inside the block to one new hash value
        size_t signatureBlockValue;
        // size_t count = 0;
        // std::cout << __LINE__ << std::endl;
        for (size_t i = 0; i < iterationSize; ++i) {
            // if (i*mShingleSize >= pSignature->size()) break;
            signatureBlockValue = (*pSignature)[i*mShingleSize];
            
            for (size_t j = 1; j < mShingleSize; ++j) {
                signatureBlockValue = mHash->hash((*pSignature)[i*mShingleSize+j]+1, signatureBlockValue+1, MAX_VALUE);
            }
            (*signature)[i] = signatureBlockValue;
            // count = i; 
        }
        // std::cout << __LINE__ << std::endl;
        if (iterationSize != mInverseIndexSize) {
            signatureBlockValue = (*pSignature)[(iterationSize+1) * mShingleSize];
            for (size_t j = 0; j < mShingleSize && j + (iterationSize+1)*mShingleSize < pSignature->size(); ++j) {
                signatureBlockValue = mHash->hash((*pSignature)[(iterationSize+1)*mShingleSize + j]+1, signatureBlockValue+1, MAX_VALUE);
            }
            // std::cout << __LINE__ << std::endl;
            // std::cout << "size: " << mInverseIndexSize << std::endl;
            // std::cout << "count: " << count << std::endl;
            (*signature)[iterationSize] = signatureBlockValue;
            // std::cout << __LINE__ << std::endl;
        }
        
        
    } else if (mShingle == 2) {
        // if 1 than take the minimum hash values of that block as the hash value
        // size_t k = 0;
        
        // while (k < mNumberOfHashFunctions*mBlockSize) {
        // // use computed hash value as a seed for the next computation
        //     size_t minValue = MAX_VALUE;
        //     for (size_t j = 0; j < mShingleSize  && k+j < mNumberOfHashFunctions*mBlockSize; ++j) {
        //         if (minValue > pSignature[k+j] ) {
        //             minValue = pSignature[k+j];
        //         }
        //     }
        //     signature->push_back(minValue);
        //     k += mShingleSize; 
        // }
    }
        // std::cout << __LINE__ << std::endl;
    
    delete pSignature;
        // std::cout << __LINE__ << std::endl;
    
    return signature; 
}

vsize_t* InverseIndex::computeSignatureWTA(const SparseMatrixFloat* pRawData, const size_t pInstance) {
    size_t sizeOfInstance = pRawData->getSizeOfInstance(pInstance);
    
    size_t mSeed = 42;
    size_t mK = mBlockSize;
    
    vsize_t* signature = new vsize_t (mNumberOfHashFunctions);;
    if (sizeOfInstance < mK) {
        mK = sizeOfInstance;
    }
    KSizeSortedMap keyValue(mK);
    
    for (size_t i = 0; i < mNumberOfHashFunctions; ++i) {
        
        for (size_t j = 0; j < sizeOfInstance; ++j) {
            size_t hashIndex = mHash->hash((pRawData->getNextElement(pInstance, j) +1), mSeed+i, MAX_VALUE);
            keyValue.insert(hashIndex, pRawData->getNextValue(pInstance, j));
        } 
        
        float maxValue = 0.0;
        size_t maxValueIndex = 0;
        for (size_t j = 0; j < mK; ++j) {
            if (keyValue.getValue(j) > maxValue) {
                maxValue = keyValue.getValue(j);
                maxValueIndex = j;
            }
        }
        (*signature)[i] = maxValueIndex;
        keyValue.clear();
    }
    if (mShingle) {
        return shingle(signature);
    }
    return signature;
}


vvsize_t_p* InverseIndex::computeSignatureVectors(const SparseMatrixFloat* pRawData) {
    if (mChunkSize <= 0) {
        mChunkSize = ceil(pRawData->size() / static_cast<float>(mNumberOfCores));
    }
    time_t timerStartCuda;
    time_t timerEndCuda;
    time_t timerStartCPU;
    time_t timerEndCPU;
    #ifdef OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(mNumberOfCores);
    omp_set_nested(1);
    #endif
    size_t cpuStart = 0;
    size_t cpuEnd = pRawData->size();
    #ifdef CUDA
    // how to split the data between cpu and gpu?
    float cpuGpuSplitFactor = 1.0;
    size_t gpuStart = 0;
    size_t gpuEnd = floor(pRawData->getNumberOfInstances() * cpuGpuSplitFactor);
    cpuStart = ceil(pRawData->getNumberOfInstances() * cpuGpuSplitFactor);
    cpuEnd = pRawData->getNumberOfInstances();
    // how many blocks, how many threads?
    size_t numberOfBlocksForGpu = 128;
    size_t numberOfThreadsForGpu = 128;
     std::cout << "gpu start: " << gpuStart << " gpuEnd: " << gpuEnd;
    std::cout << " cpu start: " << cpuStart << " cpuEnd: " << cpuEnd << std::endl;
    #endif
    
   
    vvsize_t_p* signatures = new vvsize_t_p(pRawData->size());
    #pragma omp parallel sections num_threads(mNumberOfCores)
    {
        // compute part of the signature on the gpu
        #ifdef CUDA
        #pragma omp section
        {
            time(&timerStartCuda);
            
            std::cout << "start cuda" << std::endl;
            mInverseIndexCuda->copyDataToGpu(pRawData);
            mInverseIndexCuda->computeSignaturesOnGpu(pRawData, gpuStart,
                                                    gpuEnd, gpuEnd - gpuStart, 
                                                    numberOfBlocksForGpu, 
                                                    numberOfThreadsForGpu, 
                                                    mShingleSize,
                                                    mBlockSize,
                                                    signatures);
            std::cout << "end cuda" << std::endl;
            time(&timerEndCuda);
            std::cout << "Computing signatures CUDA needs " << difftime(timerEndCuda, timerStartCuda) << " seconds." << std::endl;
        }
        #endif
        
        // compute other parts of the signature on the computed
        #pragma omp section
        {
            std::cout << "start cpu" << std::endl;
            time(&timerStartCPU);
            
            // timerStartCPU 
            #ifdef CUDA
            #pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores-1)
            #endif
            #ifndef CUDA
            #pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
            #endif
            for (size_t instance = cpuStart; instance < cpuEnd; ++instance) {
                if (mHashAlgorithm == 0) {
                    // use minHash
        // std::cout << __LINE__ << std::endl;
                    
                    (*signatures)[instance] = computeSignature(pRawData, instance);
        // std::cout << __LINE__ << std::endl;
                    
                } else if (mHashAlgorithm == 1) {
                    // use wta hash
                    (*signatures)[instance] = computeSignatureWTA(pRawData, instance);
                }
            }
            std::cout << "end cpu" << std::endl;
            time(&timerEndCPU);
            
            std::cout << "Computing signatures CPU needs " << difftime(timerEndCPU, timerStartCPU) << " seconds." << std::endl;
            
        }
    } 
    
    return signatures;
}
umap_uniqueElement* InverseIndex::computeSignatureMap(const SparseMatrixFloat* pRawData) {
    mDoubleElementsQueryCount = 0;
    const size_t sizeOfInstances = pRawData->size();
    umap_uniqueElement* instanceSignature = new umap_uniqueElement();
    instanceSignature->reserve(sizeOfInstances);
    vvsize_t_p* signatures = computeSignatureVectors(pRawData);
#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
    for (size_t i = 0; i < signatures->size(); ++i) {

        size_t signatureId = 0;
        for (size_t j = 0; j < pRawData->getSizeOfInstance(i); ++j) {
                signatureId = mHash->hash((pRawData->getNextElement(i, j) +1), (signatureId+1), MAX_VALUE);
        }
        
        if (instanceSignature->find(signatureId) == instanceSignature->end()) {
                vsize_t* doubleInstanceVector = new vsize_t(1);
                (*doubleInstanceVector)[0] = i;
                uniqueElement element;
                element.instances = doubleInstanceVector; 
                element.signature = (*signatures)[i];
                #pragma omp critical
                (*instanceSignature)[signatureId] = element;
        } else {
            #pragma omp critical
            {
                (*instanceSignature)[signatureId].instances->push_back(i);
                mDoubleElementsQueryCount += 1;
                // delete (*signatures)[i];
            }
        } 
    }
    delete signatures;
    return instanceSignature;
}
void InverseIndex::fit(const SparseMatrixFloat* pRawData) {
    mMaxNnz = pRawData->getMaxNnz();
    time_t timerStart;
    time_t timerEnd;
    // compute signatures
    time(&timerStart);
    vvsize_t_p* signatures = computeSignatureVectors(pRawData);
    time(&timerEnd);
    std::cout << "Computing signatures needs " << difftime(timerEnd, timerStart) << " seconds." << std::endl;
    time(&timerStart);
    // compute how often the inverse index should be pruned 
    size_t pruneEveryNInstances = ceil(signatures->size() * mPruneInverseIndexAfterInstance);
    #ifdef OPENMP
    omp_set_dynamic(0);
    #endif
    // store signatures in signatureStorage
#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
    for (size_t i = 0; i < signatures->size(); ++i) {

        size_t signatureId = 0;
        for (size_t j = 0; j < pRawData->getSizeOfInstance(i); ++j) {
                signatureId = mHash->hash((pRawData->getNextElement(i, j) +1), (signatureId+1), MAX_VALUE);
        }
        auto itSignatureStorage = mSignatureStorage->find(signatureId);
        if (itSignatureStorage == mSignatureStorage->end()) {
            vsize_t* doubleInstanceVector = new vsize_t(1);
            (*doubleInstanceVector)[0] = i;
            uniqueElement element;
            element.instances = doubleInstanceVector;
            element.signature = (*signatures)[i];
            #pragma omp critical
            mSignatureStorage->operator[](signatureId) = element;
        } else {
            #pragma omp critical
            {            
                mSignatureStorage->operator[](signatureId).instances->push_back(i);
                mDoubleElementsStorageCount += 1;
                // delete (*signatures)[i];
            }
        }        
        for (size_t j = 0; j < (*signatures)[i]->size(); ++j) {
            mInverseIndexStorage->insert(j, (*(*signatures)[i])[j], i, mRemoveValueWithLeastSigificantBit);
        }
        if (signatures->size() == pruneEveryNInstances) {
            pruneEveryNInstances += pruneEveryNInstances;
            if (mPruneInverseIndex > -1) {
                mInverseIndexStorage->prune(mPruneInverseIndex);
            }
            if (mRemoveHashFunctionWithLessEntriesAs > -1) {
                mInverseIndexStorage->removeHashFunctionWithLessEntriesAs(mRemoveHashFunctionWithLessEntriesAs);
            }
        }
    }
    if (mPruneInverseIndex > -1) {
        mInverseIndexStorage->prune(mPruneInverseIndex);
    }
    if (mRemoveHashFunctionWithLessEntriesAs > -1) {
        mInverseIndexStorage->removeHashFunctionWithLessEntriesAs(mRemoveHashFunctionWithLessEntriesAs);
    }
    time(&timerEnd);
    std::cout << "Inserting in inverse index needs " << difftime(timerEnd, timerStart) << " seconds." << std::endl;
    delete signatures;
}


neighborhood* InverseIndex::kneighborsCuda(const umap_uniqueElement* pSignaturesMap, 
                                        const size_t pNneighborhood, 
                                        const bool pDoubleElementsStorageCount,
                                        const size_t pNumberOfBlocksHistogram,
                                        const size_t pNumberOfThreadsHistogram,
                                        const size_t pNumberOfBlocksDistance,
                                        const size_t pNumberOfThreadsDistance,
                                        size_t pFast, size_t pDistance) {
    std::vector<vvsize_t_p*>* hitsPerInstance 
                            = new std::vector<vvsize_t_p*>(pSignaturesMap->size());                        
#ifdef OPENMP
#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
#endif 
    for (size_t i = 0; i < pSignaturesMap->size(); ++i) {
        umap_uniqueElement::const_iterator instanceId = pSignaturesMap->begin();
        std::advance(instanceId, i);
        if (instanceId == pSignaturesMap->end()) continue;
        std::unordered_map<size_t, size_t> neighborhood;
        vvsize_t_p* hits = new vvsize_t_p();
        const vsize_t* signature = instanceId->second.signature; 
        for (size_t j = 0; j < signature->size(); ++j) {
            size_t hashID = (*signature)[j];
            if (hashID != 0 && hashID != MAX_VALUE) {
                size_t collisionSize = 0; 
                
                vsize_t* instances = mInverseIndexStorage->getElement(j, hashID);
                
                if (instances == NULL) continue;
                if (instances->size() != 0) {
                    collisionSize = instances->size();
                } else { 
                    continue;
                }
                
                if (collisionSize < mMaxBinSize && collisionSize > 0) {
                    hits->push_back(instances);
                }
            }
        }
        (*hitsPerInstance)[i] = hits;
        
    }
    neighborhood* neighbors = new neighborhood();
    #ifdef CUDA    
    mInverseIndexCuda->computeHitsOnGpu(hitsPerInstance, neighbors,
                                        pNneighborhood, hitsPerInstance->size(),
                                        pNumberOfBlocksHistogram,
                                        pNumberOfThreadsHistogram,
                                        pNumberOfBlocksDistance,
                                        pNumberOfThreadsDistance,
                                        pFast, pDistance,
                                        mExcessFactor, mMaxNnz);
    #endif
    return neighbors;                                  

}
neighborhood* InverseIndex::kneighbors(const umap_uniqueElement* pSignaturesMap, 
                                        const size_t pNneighborhood, 
                                        const bool pDoubleElementsStorageCount,
                                        size_t pFast, size_t pDistance, const bool pNoneSingleInstance) {

    #ifdef CUDA
        return kneighborsCuda(pSignaturesMap, pNneighborhood, pDoubleElementsStorageCount,
                                128,128, 512, 32, pFast, pDistance);
    #endif                                       
    size_t doubleElements = 0;
    if (pNoneSingleInstance) {
        if (pDoubleElementsStorageCount) {
            doubleElements = mDoubleElementsStorageCount;
        } else {
            doubleElements = mDoubleElementsQueryCount;
        }
    }

#ifdef OPENMP
    omp_set_dynamic(0);
#endif
    vvint* neighbors = new vvint();
    vvfloat* distances = new vvfloat();
    neighbors->resize(pSignaturesMap->size()+doubleElements);
    distances->resize(pSignaturesMap->size()+doubleElements);
    if (mChunkSize <= 0) {
        mChunkSize = ceil(mInverseIndexStorage->size() / static_cast<float>(mNumberOfCores));
    }

#ifdef OPENMP
#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
#endif 

    for (size_t i = 0; i < pSignaturesMap->size(); ++i) {
        umap_uniqueElement::const_iterator instanceId = pSignaturesMap->begin();
        std::advance(instanceId, i);
        if (instanceId == pSignaturesMap->end()) continue;
        std::unordered_map<size_t, size_t> neighborhood;
        // neighborhood.reserve(mMaxBinSize*2);
        const vsize_t* signature = instanceId->second.signature; 
        for (size_t j = 0; j < signature->size(); ++j) {
            size_t hashID = (*signature)[j];
            if (hashID != 0 && hashID != MAX_VALUE) {
                size_t collisionSize = 0; 
                
                const vsize_t* instances = mInverseIndexStorage->getElement(j, hashID);
                
                if (instances == NULL) continue;
                if (instances->size() != 0) {
                    collisionSize = instances->size();
                } else { 
                    continue;
                }
                
                if (collisionSize < mMaxBinSize && collisionSize > 0) {
                    for (size_t k = 0; k < instances->size(); ++k) {
                        neighborhood[(*instances)[k]] += 1;
                    }
                } 
            }
        }

        if (neighborhood.size() == 0) {
            vint emptyVectorInt;
            emptyVectorInt.push_back(1);
            vfloat emptyVectorFloat;
            emptyVectorFloat.push_back(1);
#ifdef OPENMP
#pragma omp critical
#endif
            { // write vector to every instance with identical signatures
                if (pNoneSingleInstance) {
                    for (size_t j = 0; j < instanceId->second.instances->size(); ++j) {
                        
                        (*neighbors)[(*instanceId->second.instances)[j]] = emptyVectorInt;
                        (*distances)[(*instanceId->second.instances)[j]] = emptyVectorFloat;
                    }
                } else {
                    (*neighbors)[0] = emptyVectorInt;
                    (*distances)[0] = emptyVectorFloat;
                }
            }
            continue;
        }

        std::vector< sort_map > neighborhoodVectorForSorting;
        for (auto it = neighborhood.begin(); it != neighborhood.end(); ++it) {
            sort_map mapForSorting;
            mapForSorting.key = (*it).first;
            mapForSorting.val = (*it).second;
            neighborhoodVectorForSorting.push_back(mapForSorting);
        }
        size_t numberOfElementsToSort = pNneighborhood * mExcessFactor;
        if (numberOfElementsToSort > neighborhoodVectorForSorting.size()) {
            numberOfElementsToSort = neighborhoodVectorForSorting.size();
        }

        std::sort(neighborhoodVectorForSorting.begin(), neighborhoodVectorForSorting.end(), mapSortDescByValue);
        
        size_t sizeOfNeighborhoodAdjusted;
        if (pNneighborhood == MAX_VALUE) {
            sizeOfNeighborhoodAdjusted = std::min(static_cast<size_t>(pNneighborhood), neighborhoodVectorForSorting.size());
        } else {
 
            sizeOfNeighborhoodAdjusted = std::min(static_cast<size_t>(pNneighborhood * mExcessFactor), neighborhoodVectorForSorting.size());
            if (sizeOfNeighborhoodAdjusted == pNneighborhood * mExcessFactor 
                    && pNneighborhood * mExcessFactor < neighborhoodVectorForSorting.size()) {
                for (size_t j = sizeOfNeighborhoodAdjusted; j < neighborhoodVectorForSorting.size(); ++j) {
                    if (j + 1 < neighborhoodVectorForSorting.size() 
                            && neighborhoodVectorForSorting[j].val == neighborhoodVectorForSorting[j+1].val) {
                                ++sizeOfNeighborhoodAdjusted;
                    } else {
                        break;
                    }
                }
            }
            
        }

        size_t count = 0;
        vvint neighborsForThisInstance(instanceId->second.instances->size());
        vvfloat distancesForThisInstance(instanceId->second.instances->size());

        for (size_t j = 0; j < neighborsForThisInstance.size(); ++j) {
            vint neighborhoodVector;
            std::vector<float> distanceVector;
            for (auto it = neighborhoodVectorForSorting.begin();
                    it != neighborhoodVectorForSorting.end(); ++it) {
                neighborhoodVector.push_back((*it).key);
                distanceVector.push_back(1 - ((*it).val / static_cast<float>(mMaximalNumberOfHashCollisions)));
                ++count;
                if (count >= sizeOfNeighborhoodAdjusted) {
                    neighborsForThisInstance[j] = neighborhoodVector;
                    distancesForThisInstance[j] = distanceVector;
                    break;
                }
            }
        }

#ifdef OPENMP
#pragma omp critical
#endif

        {   // write vector to every instance with identical signatures
            if (pNoneSingleInstance) {
                for (size_t j = 0; j < instanceId->second.instances->size(); ++j) {
                    
                    (*neighbors)[(*instanceId->second.instances)[j]] = neighborsForThisInstance[j];
                    (*distances)[(*instanceId->second.instances)[j]] = distancesForThisInstance[j];
                }
            } else {
                (*neighbors)[0] = neighborsForThisInstance[0];
                (*distances)[0] = distancesForThisInstance[0];
            }
        
        }
    }

    neighborhood* neighborhood_ = new neighborhood();
    neighborhood_->neighbors = neighbors;
    neighborhood_->distances = distances;

    return neighborhood_;
    
}