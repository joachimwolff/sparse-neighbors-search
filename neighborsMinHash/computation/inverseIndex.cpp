/**
 Copyright 2015 Joachim Wolff
 Master Thesis
 Tutors: Fabrizio Costa, Milad Miladi
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
#include "kSizeSortedMap.h"


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
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance, int pRemoveHashFunctionWithLessEntriesAs,
                    size_t pHashAlgorithm, size_t pBlockSize, size_t pShingle, size_t pRemoveValueWithLeastSigificantBit) {   
        std::cout << __LINE__ << std::endl;
                        
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
    size_t maximalFeatures = 5000;
    size_t inverseIndexSize;
    if (mShingle == 0) {
        if (mBlockSize == 0) {
            mBlockSize = 1;
        }
        inverseIndexSize = mNumberOfHashFunctions * mBlockSize;
    } else {
        inverseIndexSize = ceil(((float) (mNumberOfHashFunctions * mBlockSize) / (float) mShingleSize));        
    }
    if (pBloomierFilter) {
        mInverseIndexStorage = new InverseIndexStorageBloomierFilter(inverseIndexSize, mMaxBinSize, maximalFeatures);
    } else {
        mInverseIndexStorage = new InverseIndexStorageUnorderedMap(inverseIndexSize, mMaxBinSize);
    }
    mRemoveValueWithLeastSigificantBit = pRemoveValueWithLeastSigificantBit;
        // std::cout << __LINE__ << std::endl;
    
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
vsize_t InverseIndex::computeSignature(const SparseMatrixFloat* pRawData, const size_t pInstance) {

    vsize_t signature(mNumberOfHashFunctions * mBlockSize);

    for(size_t j = 0; j < mNumberOfHashFunctions; ++j) {
        for (size_t k = 0; k < mBlockSize; ++k) {
            size_t minHashValue = MAX_VALUE;        
            for (size_t i = 0; i < pRawData->getSizeOfInstance(pInstance); ++i) {
                size_t hashValue = mHash->hash((pRawData->getNextElement(pInstance, i) +1), (j+1+k), MAX_VALUE);
                if (hashValue < minHashValue) {
                    minHashValue = hashValue;
                }
            }
            signature[j+k] = minHashValue;
        }
       
    }
    // reduce number of hash values by a factor of mShingleSize
        // std::cout << __LINE__ << std::endl;
    
    if (mShingle) {
        return shingle(signature);
    }
    
    return signature;
}

vsize_t InverseIndex::shingle(vsize_t pSignature) {
    vsize_t signature;
        // std::cout << "BUUUUU" << std::endl;
    
    if (mShingle == 1) {
        // if 0 than combine hash values inside the block to one new hash value
        size_t k = 0;
        // std::cout << __LINE__ << std::endl;
        while (k < mNumberOfHashFunctions*mBlockSize && k < pSignature.size()) {
        // use computed hash value as a seed for the next computation
        // std::cout << __LINE__ << std::endl;
            size_t signatureBlockValue = pSignature[k];
            for (size_t j = 0; j < mShingleSize && k+j < mNumberOfHashFunctions*mBlockSize; ++j) {
                signatureBlockValue = mHash->hash(pSignature[k+j]+1,  signatureBlockValue+1, MAX_VALUE);
            }
            signature.push_back(signatureBlockValue);
            k += mShingleSize; 
            // std::cout << __LINE__ << std::endl;
            
        }
        // std::cout << "Size of signature: " << signature.size() << std::endl;
        
    } else if (mShingle == 2) {
        // if 1 than take the minimum hash values of that block as the hash value
        size_t k = 0;
        
        while (k < mNumberOfHashFunctions*mBlockSize) {
        // use computed hash value as a seed for the next computation
            size_t minValue = MAX_VALUE;
            for (size_t j = 0; j < mShingleSize  && k+j < mNumberOfHashFunctions*mBlockSize; ++j) {
                if (minValue > pSignature[k+j] ) {
                    minValue = pSignature[k+j];
                }
            }
            signature.push_back(minValue);
            k += mShingleSize; 
        }
    }
    return signature; 
}

vsize_t InverseIndex::computeSignatureWTA(const SparseMatrixFloat* pRawData, const size_t pInstance) {
    size_t sizeOfInstance = pRawData->getSizeOfInstance(pInstance);
    vsize_t signatureHash(sizeOfInstance);
    
    // signatureHash.reserve(sizeOfInstance);    
    size_t mSeed = 42;
    size_t mK = mBlockSize;
    // mNumberOfHashFunctions = 800;
    
    KSizeSortedMap keyValue(mK);
    vsize_t signature(mNumberOfHashFunctions);
    if (sizeOfInstance < mK) {
        mK = sizeOfInstance;
    }
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
        signature[i] = maxValueIndex;
        // keyValue.clear();
        // signatureHash.clear();
    }
    if (mShingle) {
        return shingle(signature);
    }
    return signature;
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
                (*instanceSignature)[signatureId] = (*mSignatureStorage)[signatureId];
                (*instanceSignature)[signatureId].instances.push_back(index);
                mDoubleElementsQueryCount += (*mSignatureStorage)[signatureId].instances.size();
            }
            continue;
        }

        // for every hash function: compute the hash values of all features and take the minimum of these
        // as the hash value for one hash function --> h_j(x) = argmin (x_i of x) f_j(x_i)
        vsize_t signature;
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
                vsize_t doubleInstanceVector(1);
                doubleInstanceVector[0] = index;
                uniqueElement element;
                element.instances = doubleInstanceVector;
                element.signature = signature;
                (*instanceSignature)[signatureId] = element;
            } else {
                (*instanceSignature)[signatureId].instances.push_back(index);
                mDoubleElementsQueryCount += 1;
            }
        }
    }
    return instanceSignature;
}
void InverseIndex::fit(const SparseMatrixFloat* pRawData) {
    // std::cout << __LINE__ << std::endl;
    
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
        vsize_t signature;
        auto itSignatureStorage = mSignatureStorage->find(signatureId);
        if (itSignatureStorage == mSignatureStorage->end()) {
            if (mHashAlgorithm == 0) {
                // use minHash
                signature = computeSignature(pRawData, index);
            } else if (mHashAlgorithm == 1) {
                // use wta hash
                        // std::cout << __LINE__ << std::endl;

                signature = computeSignatureWTA(pRawData, index);
            }
        } else {
            signature = itSignatureStorage->second.signature;
        }
#ifdef OPENMP
#pragma omp critical
#endif
        {   
                    // std::cout << __LINE__ << std::endl;

            ++pruneCount;
            if (itSignatureStorage == mSignatureStorage->end()) {
                vsize_t doubleInstanceVector(1);
                doubleInstanceVector[0] = index;
                uniqueElement element;
                element.instances = doubleInstanceVector;
                element.signature = signature;
                mSignatureStorage->operator[](signatureId) = element;
            } else {
                 mSignatureStorage->operator[](signatureId).instances.push_back(index);
                 mDoubleElementsStorageCount += 1;
            }
        }
            // std::cout << __LINE__ << std::endl;
        
        for (size_t j = 0; j < signature.size(); ++j) {
            mInverseIndexStorage->insert(j, signature[j], index, mRemoveValueWithLeastSigificantBit);
        }
            // std::cout << __LINE__ << std::endl;
        
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
    // std::cout << __LINE__ << std::endl;
    
}

neighborhood* InverseIndex::kneighbors(const umap_uniqueElement* pSignaturesMap, 
                                        const size_t pNneighborhood, const bool pDoubleElementsStorageCount) {
    // std::cout << __LINE__ << std::endl;
                                            
    size_t doubleElements = 0;
    if (pDoubleElementsStorageCount) {
        doubleElements = mDoubleElementsStorageCount;
    } else {
        doubleElements = mDoubleElementsQueryCount;
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
    // std::cout << __LINE__ << std::endl;

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
    // std::cout << __LINE__ << std::endl;
        
        umap_uniqueElement::const_iterator instanceId = pSignaturesMap->begin();
        std::advance(instanceId, i); 
    // std::cout << __LINE__ << std::endl;
        
        std::unordered_map<size_t, size_t> neighborhood;
        const vsize_t signature = instanceId->second.signature;
    // std::cout << __LINE__ << std::endl;
        
        for (size_t j = 0; j < signature.size(); ++j) {
            size_t hashID = signature[j];
    // std::cout << __LINE__ << std::endl;
            
            if (hashID != 0 && hashID != MAX_VALUE) {
                size_t collisionSize = 0;
    // std::cout << __LINE__ << std::endl;

                const vsize_t* instances = mInverseIndexStorage->getElement(j, hashID);
    // std::cout << __LINE__ << std::endl;
                if (instances == NULL) continue;
                if (instances->size() != 0) {
                    collisionSize = instances->size();
                } else { 
                    continue;
                }
    // std::cout << __LINE__ << std::endl;
                
                if (collisionSize < mMaxBinSize && collisionSize > 0) {
                    for (size_t k = 0; k < instances->size(); ++k) {
                        neighborhood[(*instances)[k]] += 1;
                    }
                }
            }
        }
    // std::cout << __LINE__ << std::endl;
        
        if (neighborhood.size() == 0) {
            vint emptyVectorInt;
            emptyVectorInt.push_back(1);
            vfloat emptyVectorFloat;
            emptyVectorFloat.push_back(1);
            for (size_t j = 0; j < instanceId->second.instances.size(); ++j) {
                (*neighbors)[instanceId->second.instances[j]] = emptyVectorInt;
                (*distances)[instanceId->second.instances[j]] = emptyVectorFloat;
            }
            continue;
        } 
    // std::cout << __LINE__ << std::endl;
        
        std::vector< sort_map > neighborhoodVectorForSorting;
        for (auto it = neighborhood.begin(); it != neighborhood.end(); ++it) {
            sort_map mapForSorting;
            mapForSorting.key = (*it).first;
            mapForSorting.val = (*it).second;
            neighborhoodVectorForSorting.push_back(mapForSorting);
        }
    // std::cout << __LINE__ << std::endl;

        size_t numberOfElementsToSort = pNneighborhood;
        if (pNneighborhood > neighborhoodVectorForSorting.size()) {
            numberOfElementsToSort = neighborhoodVectorForSorting.size();
        }
        std::partial_sort(neighborhoodVectorForSorting.begin(), neighborhoodVectorForSorting.begin()+numberOfElementsToSort, neighborhoodVectorForSorting.end(), mapSortDescByValue);
        size_t sizeOfNeighborhoodAdjusted;
    // std::cout << __LINE__ << std::endl;
        
        if (pNneighborhood == MAX_VALUE) {
            sizeOfNeighborhoodAdjusted = std::min(static_cast<size_t>(pNneighborhood), neighborhoodVectorForSorting.size());
        } else {
            sizeOfNeighborhoodAdjusted = std::min(static_cast<size_t>(pNneighborhood * mExcessFactor), neighborhoodVectorForSorting.size());
        }
        size_t count = 0;
    // std::cout << __LINE__ << std::endl;
        
        vvint neighborsForThisInstance(instanceId->second.instances.size());
        vvfloat distancesForThisInstance(instanceId->second.instances.size());

        for (size_t j = 0; j < neighborsForThisInstance.size(); ++j) {
            vint neighborhoodVector;
            std::vector<float> distanceVector;
            if (neighborhoodVectorForSorting[0].key != instanceId->second.instances[j]) {
                neighborhoodVector.push_back(instanceId->second.instances[j]);
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

        {     
                // std::cout << __LINE__ << std::endl;
            for (size_t j = 0; j < instanceId->second.instances.size(); ++j) {
                (*neighbors)[instanceId->second.instances[j]] = neighborsForThisInstance[j];
                (*distances)[instanceId->second.instances[j]] = distancesForThisInstance[j];
            }
        }
        // delete neighborsForThisInstance;
        // delete distancesForThisInstance;
    }
    // std::cout << __LINE__ << std::endl;
    
    neighborhood* neighborhood_ = new neighborhood();
    neighborhood_->neighbors = neighbors;
    neighborhood_->distances = distances;
    // std::cout << __LINE__ << std::endl;
      
    return neighborhood_;
}