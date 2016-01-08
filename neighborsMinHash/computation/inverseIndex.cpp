/**
 Copyright 2015 Joachim Wolff
 Master Thesis
 Tutor: Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/

#include <algorithm>
#include <iostream>
#include <iterator>
#include <utility>

#ifdef OPENMP
#include <omp.h>
#endif

#include "inverseIndex.h"

class sort_map {
  public:
    size_t key;
    size_t val;
};

bool mapSortDescByValue(const sort_map& a, const sort_map& b) {
        return a.val > b.val;
};

InverseIndex::InverseIndex(size_t pNumberOfHashFunctions, size_t pShingleSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions, size_t pBloomierFilter,
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance, int pRemoveHashFunctionWithLessEntriesAs) {   
                        
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
    mSignatureStorage = new umap_uniqueElement();
    mHash = new Hash();
    size_t maximalFeatures = 5000;
    
    size_t inverseIndexSize = ceil(((float) mNumberOfHashFunctions / (float) mShingleSize)+1);
    if (pBloomierFilter) {
        // std::cout << "Using bloomier filter. " << std::endl;
        mInverseIndexStorage = new InverseIndexStorageBloomierFilter(inverseIndexSize, mMaxBinSize, maximalFeatures);
    } else {
        // std::cout << "Using unorderedMap. " << std::endl;
        
        mInverseIndexStorage = new InverseIndexStorageUnorderedMap(inverseIndexSize, mMaxBinSize);
    }
}
 
InverseIndex::~InverseIndex() {

    for (auto it = (*mSignatureStorage).begin(); it != (*mSignatureStorage).end(); ++it) {
        delete it->second->instances;
        delete it->second->signature;
        delete it->second;

    }
    delete mSignatureStorage;
    delete mHash;
}

std::map<size_t, size_t>* InverseIndex::getDistribution() {
    return mInverseIndexStorage->getDistribution();
}

 // compute the signature for one instance
vsize_t* InverseIndex::computeSignature(const SparseMatrixFloat* pRawData, const size_t pInstance) {

    vsize_t signatureHash; 
    signatureHash.reserve(mNumberOfHashFunctions);

    for(size_t j = 0; j < mNumberOfHashFunctions; ++j) {
        size_t minHashValue = MAX_VALUE;
        for (size_t i = 0; i < pRawData->getSizeOfInstance(pInstance); ++i) {
            //  hash(size_t pKey, size_t pModulo, size_t pSeed)
            size_t hashValue = mHash->hash((pRawData->getNextElement(pInstance, i) +1), (j+1), MAX_VALUE);
            if (hashValue < minHashValue) {
                minHashValue = hashValue;
            }
        }
        signatureHash[j] = minHashValue;
    }
    // reduce number of hash values by a factor of ShingleSize
    size_t k = 0;
    vsize_t* signature = new vsize_t();
    signature->reserve((mNumberOfHashFunctions / mShingleSize) + 1);
    while (k < (mNumberOfHashFunctions)) {
        // use computed hash value as a seed for the next computation
        size_t signatureBlockValue = signatureHash[k];
        for (size_t j = 0; j < mShingleSize; ++j) {
            signatureBlockValue = mHash->hash((signatureHash[k+j]),  signatureBlockValue, MAX_VALUE);
        }
        signature->push_back(signatureBlockValue);
        k += mShingleSize; 
    }
    
    
    return signature;
}
umap_uniqueElement* InverseIndex::computeSignatureMap(const SparseMatrixFloat* pRawData) {
    mDoubleElementsQueryCount = 0;
    const size_t sizeOfInstances = pRawData->size();
    umap_uniqueElement* instanceSignature = new umap_uniqueElement();
    (*instanceSignature).reserve(sizeOfInstances);
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
        // vsize_t* features = pRawData->getFeatureRow(index);
        // compute unique id
        size_t signatureId = 0;
        for (size_t j = 0; j < pRawData->getSizeOfInstance(index); ++j) {
                signatureId = mHash->hash((pRawData->getNextElement(index, j) +1), (signatureId+1), MAX_VALUE);
        }
        // signature is in storage && 
        auto signatureIt = (*mSignatureStorage).find(signatureId);
        if (signatureIt != (*mSignatureStorage).end() && (instanceSignature->find(signatureId) != instanceSignature->end())) {
#ifdef OPENMP
#pragma omp critical
#endif
            {
                instanceSignature->operator[](signatureId) = (*mSignatureStorage)[signatureId];
                instanceSignature->operator[](signatureId)->instances->push_back(index);
                mDoubleElementsQueryCount += (*mSignatureStorage)[signatureId]->instances->size();
            }
            continue;
        }

        // for every hash function: compute the hash values of all features and take the minimum of these
        // as the hash value for one hash function --> h_j(x) = argmin (x_i of x) f_j(x_i)
        vsize_t* signature = computeSignature(pRawData, index);
#ifdef OPENMP
#pragma omp critical
#endif
        {
            if (instanceSignature->find(signatureId) == instanceSignature->end()) {
                vsize_t* doubleInstanceVector = new vsize_t(1);
                (*doubleInstanceVector)[0] = index;
                uniqueElement* element = new uniqueElement();;
                element->instances = doubleInstanceVector;
                element->signature = signature;
                instanceSignature->operator[](signatureId) = element;
            } else {
                instanceSignature->operator[](signatureId)->instances->push_back(index);
                mDoubleElementsQueryCount += 1;
            }
        }
    }
    return instanceSignature;
}
void InverseIndex::fit(const SparseMatrixFloat* pRawData) {
    size_t pruneEveryNIterations = pRawData->size() * mPruneInverseIndexAfterInstance;
    size_t pruneCount = 0;
    mDoubleElementsStorageCount = 0;
    if (mChunkSize <= 0) {
        mChunkSize = ceil(pRawData->size() / static_cast<float>(mNumberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
#ifdef OPENMP
#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
#endif

    for (size_t index = 0; index < pRawData->size(); ++index) {
        size_t signatureId = 0;
        for (size_t j = 0; j < pRawData->getSizeOfInstance(index); ++j) {
            signatureId = mHash->hash((pRawData->getNextElement(index, j) +1), (signatureId+1), MAX_VALUE);
        }
        vsize_t* signature;
        auto itSignatureStorage = mSignatureStorage->find(signatureId);
        if (itSignatureStorage == mSignatureStorage->end()) {
            signature = computeSignature(pRawData, index);
        } else {
            signature = itSignatureStorage->second->signature;
        }
#ifdef OPENMP
#pragma omp critical
#endif
        {   
            ++pruneCount;
            if (itSignatureStorage == mSignatureStorage->end()) {
                vsize_t* doubleInstanceVector = new vsize_t(1);
                (*doubleInstanceVector)[0] = index;
                uniqueElement* element = new uniqueElement();
                element->instances = doubleInstanceVector;
                element->signature = signature;
                mSignatureStorage->operator[](signatureId) = element;
            } else {
                 mSignatureStorage->operator[](signatureId)->instances->push_back(index);
                 mDoubleElementsStorageCount += 1;
            }
        }
        
        for (size_t j = 0; j < signature->size(); ++j) {
            mInverseIndexStorage->insert(j, (*signature)[j], index);
        }
        if (mPruneInverseIndexAfterInstance > 0) {
#ifdef OPENMP
#pragma omp critical
#endif
            {
                if (pruneCount >= pruneEveryNIterations) {
                    pruneCount = 0;
                    
                    if (mPruneInverseIndex > 0) {
                        mInverseIndexStorage->prune(mPruneInverseIndex);
                    }
                    if (mRemoveHashFunctionWithLessEntriesAs >= 0) {
                        mInverseIndexStorage->removeHashFunctionWithLessEntriesAs(mRemoveHashFunctionWithLessEntriesAs);
                    }
                }
            }           
        }
    }
    if (mPruneInverseIndex > 0) {
        mInverseIndexStorage->prune(mPruneInverseIndex);
    }
    if (mRemoveHashFunctionWithLessEntriesAs >= 0) {
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
        
        std::unordered_map<size_t, size_t> neighborhood;
        const vsize_t* signature = instanceId->second->signature;
        for (size_t j = 0; j < signature->size(); ++j) {
            size_t hashID = (*signature)[j];
            if (hashID != 0 && hashID != MAX_VALUE) {
                size_t collisionSize = 0;

                const vsize_t* instances = mInverseIndexStorage->getElement(j, hashID);
                if (instances != NULL && instances->size() != 0) {
                    collisionSize = instances->size();
                } else { 
                    continue;
                }
                
                if (collisionSize < mMaxBinSize && collisionSize > 0) {
                    for (size_t k = 0; k < instances->size(); ++k) {
                        neighborhood[instances->at(k)] += 1;
                    }
                }
            }
        }
        if (neighborhood.size() == 0) {
            vint emptyVectorInt;
            emptyVectorInt.push_back(1);
            vfloat emptyVectorFloat;
            emptyVectorFloat.push_back(1);
            for (size_t j = 0; j < instanceId->second->instances->size(); ++j) {
                (*neighbors)[instanceId->second->instances->operator[](j)] = emptyVectorInt;
                (*distances)[instanceId->second->instances->operator[](j)] = emptyVectorFloat;
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

        size_t numberOfElementsToSort = pNneighborhood;
        if (pNneighborhood > neighborhoodVectorForSorting.size()) {
            numberOfElementsToSort = neighborhoodVectorForSorting.size();
        }
        std::partial_sort(neighborhoodVectorForSorting.begin(), neighborhoodVectorForSorting.begin()+numberOfElementsToSort, neighborhoodVectorForSorting.end(), mapSortDescByValue);
        size_t sizeOfNeighborhoodAdjusted;
        if (pNneighborhood == MAX_VALUE) {
            sizeOfNeighborhoodAdjusted = std::min(static_cast<size_t>(pNneighborhood), neighborhoodVectorForSorting.size());
        } else {
            sizeOfNeighborhoodAdjusted = std::min(static_cast<size_t>(pNneighborhood * mExcessFactor), neighborhoodVectorForSorting.size());
        }
        size_t count = 0;
        vvint* neighborsForThisInstance = new vvint(instanceId->second->instances->size());
        vvfloat* distancesForThisInstance = new vvfloat(instanceId->second->instances->size());

        for (size_t j = 0; j < neighborsForThisInstance->size(); ++j) {
            vint neighborhoodVector;
            std::vector<float> distanceVector;
            if (neighborhoodVectorForSorting[0].key != instanceId->second->instances->operator[](j)) {
                neighborhoodVector.push_back(instanceId->second->instances->operator[](j));
                distanceVector.push_back(0);
                ++count;
            }
            for (auto it = neighborhoodVectorForSorting.begin();
                    it != neighborhoodVectorForSorting.end(); ++it) {
                neighborhoodVector.push_back((*it).key);
                distanceVector.push_back(1 - ((*it).val / static_cast<float>(mMaximalNumberOfHashCollisions)));
                ++count;
                if (count >= sizeOfNeighborhoodAdjusted) {
                    (*neighborsForThisInstance)[j] = neighborhoodVector;
                    (*distancesForThisInstance)[j] = distanceVector;
                    break;
                }
            }
        }
#ifdef OPENMP
#pragma omp critical
#endif
        {     
            for (size_t j = 0; j < instanceId->second->instances->size(); ++j) {
                (*neighbors)[instanceId->second->instances->operator[](j)] = (*neighborsForThisInstance)[j];
                (*distances)[instanceId->second->instances->operator[](j)] = (*distancesForThisInstance)[j];
            }
        }
    }
    neighborhood* neighborhood_ = new neighborhood();
    neighborhood_->neighbors = neighbors;
    neighborhood_->distances = distances;
    return neighborhood_;
}