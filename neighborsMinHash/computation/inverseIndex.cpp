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
    } else {
        mInverseIndexSize = ceil(((float) (mNumberOfHashFunctions * mBlockSize) / (float) mShingleSize));        
    }
        mInverseIndexStorage = new InverseIndexStorageUnorderedMap(mInverseIndexSize, mMaxBinSize);
    mRemoveValueWithLeastSigificantBit = pRemoveValueWithLeastSigificantBit;
    #ifdef CUDA
    mInverseIndexCuda = new InverseIndexCuda(pNumberOfHashFunctions, mShingle, mShingleSize, mBlockSize);
    #endif
}
 
InverseIndex::~InverseIndex() {
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
    
    vsize_t* signature = new vsize_t(mNumberOfHashFunctions*mBlockSize / mShingleSize);
    
    if (mShingle == 1) {
        
        // if 0 than combine hash values inside the block to one new hash value
        size_t signatureBlockValue;
        size_t count = 0;
        
        for (size_t i = 0; i < signature->size(); ++i) {
            if (i*mShingleSize >= pSignature->size()) break;
            signatureBlockValue = (*pSignature)[i*mShingleSize];
            
            for (size_t j = 1; j < mShingleSize; ++j) {
                signatureBlockValue = mHash->hash((*pSignature)[i*mShingleSize+j]+1, signatureBlockValue+1, MAX_VALUE);
            }
            (*signature)[i] = signatureBlockValue;
            count = i;
        }
        
        signatureBlockValue = (*pSignature)[count*mShingleSize];
        for (size_t j = count; count * mShingleSize + j < pSignature->size(); ++j) {
            signatureBlockValue = mHash->hash((*pSignature)[count * mShingleSize + j]+1, signatureBlockValue+1, MAX_VALUE);
        }
        
        (*signature)[count+1] = signatureBlockValue;
        
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
    delete pSignature;
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
    std::cout << __LINE__ << std::endl;
    
    vvsize_t_p* signatures = new vvsize_t_p();
    // omp_set_nested(1);
    omp_set_dynamic(0);
    omp_set_num_threads(mNumberOfCores);
    omp_set_nested(1);
    size_t cpuStart = 0;
    size_t cpuEnd = pRawData->size();
    #ifdef CUDA
    // how to split the data between cpu and gpu?
    float cpuGpuSplitFactor = 0.7;
    size_t gpuStart = 0;
    size_t gpuEnd = floor(pRawData->getNumberOfInstances() * cpuGpuSplitFactor);
    cpuStart = ceil(pRawData->getNumberOfInstances() * cpuGpuSplitFactor);
    cpuEnd = pRawData->getNumberOfInstances();
    // how many blocks, how many threads?
    size_t numberOfBlocksForGpu = 64;
    size_t numberOfThreadsForGpu = 128;
    std::cout << __LINE__ << std::endl;
    #endif
    
    #pragma omp parallel num_threads(mNumberOfCores)
    {
        vvsize_t_p* signaturesPerThread = new vvsize_t_p();
        // compute part of the signature on the gpu
        #ifdef CUDA
        #pragma omp single nowait
        {
            mInverseIndexCuda->copyDataToGpu(pRawData);
            signaturesPerThread = mInverseIndexCuda->computeSignaturesOnGpu(pRawData, gpuStart,
                                                                        gpuEnd, gpuEnd - gpuStart, 
                                                                        numberOfBlocksForGpu, 
                                                                        numberOfThreadsForGpu);
        }
        #endif
        // compute other parts of the signature on the computed   
        #pragma omp for schedule(static, mChunkSize)
        for (size_t instance = cpuStart; instance < cpuEnd; ++instance) {
            if (mHashAlgorithm == 0) {
                // use minHash
                signaturesPerThread->push_back(computeSignature(pRawData, instance));
            } else if (mHashAlgorithm == 1) {
                // use wta hash
                signaturesPerThread->push_back(computeSignatureWTA(pRawData, instance));
            }
        }
        #pragma omp barrier
        #pragma omp for schedule(static) ordered
        for(int i=0; i < omp_get_num_threads(); i++) {
            #pragma omp ordered
            signatures->insert(signatures->end(), signaturesPerThread->begin(), signaturesPerThread->end());
        }       
        
    } 
    return signatures;
}
umap_uniqueElement* InverseIndex::computeSignatureMap(const SparseMatrixFloat* pRawData) {
    mDoubleElementsQueryCount = 0;
    const size_t sizeOfInstances = pRawData->size();
    umap_uniqueElement* instanceSignature = new umap_uniqueElement();
    instanceSignature->reserve(sizeOfInstances);
    if (mChunkSize <= 0) {
        mChunkSize = ceil(pRawData->size() / static_cast<float>(mNumberOfCores));
    }
#ifdef OPENMP 
    omp_set_dynamic(0);
#endif

#ifdef OPENMP
#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
#endif
    for(size_t index = 0; index < pRawData->size(); ++index) {
        // compute unique id
        size_t signatureId = 0;
        for (size_t j = 0; j < pRawData->getSizeOfInstance(index); ++j) {
                signatureId = mHash->hash((pRawData->getNextElement(index, j) +1), (signatureId+1), MAX_VALUE);
        }
        auto signatureIt = (*mSignatureStorage).find(signatureId);
        if (signatureIt != (*mSignatureStorage).end() && (instanceSignature->find(signatureId) != instanceSignature->end())) {
#ifdef OPENMP
#pragma omp critical
#endif
            {
                (*instanceSignature)[signatureId] = (*mSignatureStorage)[signatureId];
                (*instanceSignature)[signatureId].instances->push_back(index);
                mDoubleElementsQueryCount += (*mSignatureStorage)[signatureId].instances->size();
            }
            continue;
        }

        // for every hash function: compute the hash values of all features and take the minimum of these
        // as the hash value for one hash function --> h_j(x) = argmin (x_i of x) f_j(x_i)
        vsize_t* signature;
        if (mHashAlgorithm == 0) {
            // use minHash
            signature = computeSignature(pRawData, index);
        } else if (mHashAlgorithm == 1) {
            // use wta hash
            signature = computeSignatureWTA(pRawData, index);
        }
#ifdef OPENMP
#pragma omp critical
#endif
        {
            if (instanceSignature->find(signatureId) == instanceSignature->end()) {
                vsize_t* doubleInstanceVector = new vsize_t(1);
                (*doubleInstanceVector)[0] = index;
                uniqueElement element;
                element.instances = doubleInstanceVector; 
                element.signature = signature;
                (*instanceSignature)[signatureId] = element;
            } else {
                (*instanceSignature)[signatureId].instances->push_back(index);
                mDoubleElementsQueryCount += 1;
            } 
        }
    }
    return instanceSignature;
}
void InverseIndex::fit(const SparseMatrixFloat* pRawData) {
    // compute signatures
    vvsize_t_p* signatures = computeSignatureVectors(pRawData);
    // compute how often the inverse index should be pruned 
    size_t pruneEveryNInstances = ceil(signatures->size() * mPruneInverseIndexAfterInstance);
    omp_set_dynamic(0);
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
}


neighborhood* InverseIndex::kneighbors(const umap_uniqueElement* pSignaturesMap, 
                                        const size_t pNneighborhood, const bool pDoubleElementsStorageCount) {

    size_t doubleElements = 0;
    if (pDoubleElementsStorageCount) {
        doubleElements = mDoubleElementsStorageCount;
    } else {
        doubleElements = mDoubleElementsQueryCount;
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

                for (size_t j = 0; j < instanceId->second.instances->size(); ++j) {
                    (*neighbors)[(*instanceId->second.instances)[j]] = emptyVectorInt;
                    (*distances)[(*instanceId->second.instances)[j]] = emptyVectorFloat;
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
            if (neighborhoodVectorForSorting[0].key != (*instanceId->second.instances)[j]) {
                neighborhoodVector.push_back((*instanceId->second.instances)[j]);
                distanceVector.push_back(0);
                ++count;
            }
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
       
            for (size_t j = 0; j < instanceId->second.instances->size(); ++j) {
                (*neighbors)[(*instanceId->second.instances)[j]] = neighborsForThisInstance[j];
                (*distances)[(*instanceId->second.instances)[j]] = distancesForThisInstance[j];
            }
        
        }
    }
    
    neighborhood* neighborhood_ = new neighborhood();
    neighborhood_->neighbors = neighbors;
    neighborhood_->distances = distances;
    return neighborhood_;
}