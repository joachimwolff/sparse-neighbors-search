/**
 Copyright 2016, 2017, 2018, 2019, 2020 Joachim Wolff
 PhD Thesis

 Copyright 2015, 2016 Joachim Wolff
 Master Thesis
 Tutor: Fabrizio Costa
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

#include <assert.h>
#ifdef OPENMP
#include <omp.h>
#endif
#include <stdlib.h>
#include <time.h>
#include "inverseIndex.h"
#include "kSizeSortedMap.h"
#include "sseExtension.h"
// #include "avxExtension.h"

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
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions,
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance,
                    int pRemoveHashFunctionWithLessEntriesAs, size_t pHashAlgorithm,
                    size_t pBlockSize, size_t pShingle,
                    size_t pRemoveValueWithLeastSigificantBit,
                    float pCpuGpuLoadBalancing, size_t pGpuHash, size_t pRangeK_Wta) {   
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
    mCpuGpuLoadBalancing = pCpuGpuLoadBalancing;
    mRangeK_Wta = pRangeK_Wta;
    mGpuHash = pGpuHash;
    if (mShingle == 0) {
        if (mBlockSize == 0) {
            mBlockSize = 1;
        }
        mInverseIndexSize = mNumberOfHashFunctions * mBlockSize;
        mShingleSize = 1;
        mBlockSize = 1;
    } else {
        mInverseIndexSize = ceil(((float) (mNumberOfHashFunctions * mBlockSize) / (float) mShingleSize));
    }
    
    mInverseIndexStorage = new InverseIndexStorageUnorderedMap(mInverseIndexSize, mMaxBinSize);
    mRemoveValueWithLeastSigificantBit = pRemoveValueWithLeastSigificantBit;
    #ifdef CUDA
    mInverseIndexCuda = new InverseIndexCuda(pNumberOfHashFunctions, mShingle,
                                             mShingleSize, mBlockSize, 
                                             mHashAlgorithm);
    #endif
}
 
InverseIndex::~InverseIndex() {
    for (auto it = mSignatureStorage->begin(); it != mSignatureStorage->end(); ++it) {
            delete (*it).second.instances;
            delete (*it).second.signature;
    }
    delete mSignatureStorage;
    delete mHash;
    delete mInverseIndexStorage;
} 

distributionInverseIndex* InverseIndex::getDistribution() {
    return mInverseIndexStorage->getDistribution();
}

// compute the signature for one instance with SSE support
vsize_t* InverseIndex::computeSignatureSSE(SparseMatrixFloat* pRawData, const size_t pInstance) {

    if (pRawData == NULL) return NULL;

    vsize_t* signature = new vsize_t(mNumberOfHashFunctions * mBlockSize);
    __m128i minimumVector;
    __m128i seed;
    __m128i argmin;
    __m128i value;
    __m128i hashValue;

    for(size_t j = 0; j < mNumberOfHashFunctions * mBlockSize; ++j) {

            minimumVector = _mm_set_epi32(MAX_VALUE, MAX_VALUE, MAX_VALUE, MAX_VALUE);

            argmin = _mm_set_epi32(0,0,0,0);

            seed = _mm_set_epi32(j+1, j+1, j+1, j+1);

            if (pRawData->getSizeOfInstance(pInstance) > 4) {
                for (size_t i = 0; i < pRawData->getSizeOfInstance(pInstance) - 4; i+=4) {
                    
                    // value = _mm_setr_epi32((pRawData->getNextElement(pInstance, i) +1), 
                    //                     (pRawData->getNextElement(pInstance, i+1) +1),
                    //                     (pRawData->getNextElement(pInstance, i+2) +1),
                    //                     (pRawData->getNextElement(pInstance, i+3) +1));
                    value = _mm_setr_epi32(((pRawData->getNextElement(pInstance, i) + 1) % (MAX_VALUE)), 
                                        ((pRawData->getNextElement(pInstance, i+1) +1) % (MAX_VALUE)),
                                        ((pRawData->getNextElement(pInstance, i+2) +1) % (MAX_VALUE)),
                                        ((pRawData->getNextElement(pInstance, i+3) +1) % (MAX_VALUE)));
                    hashValue = mHash->hash_SSE(value, seed);
                    
                    minimumVector = _mm_min_epu32(hashValue, minimumVector);

                    // compare all four hash values and store minimum for each element
                    argmin = _mm_argmin_change_epi32(argmin, minimumVector, hashValue, value);

                }
                (*signature)[j] = _mm_get_argmin(argmin, minimumVector);
            } else {
                uint64_t nearestNeighborsValue = MAX_VALUE;    
                size_t argmin_local = 0;
                size_t seed_local = 0;
                seed_local = j + 1;    
                for (size_t i = 0; i < pRawData->getSizeOfInstance(pInstance); i++) {
                    uint64_t hashValue = mHash->hash(((pRawData->getNextElement(pInstance, i) +1) % MAX_VALUE), seed_local, MAX_VALUE);
                
                    if (hashValue < nearestNeighborsValue) {
                        nearestNeighborsValue = hashValue;
                        argmin_local = pRawData->getNextElement(pInstance, i);
                    }
                }
                (*signature)[j] = argmin_local;
            }
    }
    // reduce number of hash values by a factor of mShingleSize
    if (mShingle) {

        return shingle(signature);
    }

    return signature;
}  

// compute the signature for one instance with SSE support
// vsize_t* InverseIndex::computeSignatureAVX(SparseMatrixFloat* pRawData, const size_t pInstance) {

//     if (pRawData == NULL) return NULL;

//     vsize_t* signature = new vsize_t(mNumberOfHashFunctions * mBlockSize);
//     __m256i minimumVector;
//     __m256i seed;
//     __m256i argmin;
//     __m256i value;
//     __m256i hashValue;

//     for(size_t j = 0; j < mNumberOfHashFunctions * mBlockSize; ++j) {

//             // minimumVector = _mm256_set_epi64(MAX_VALUE, MAX_VALUE, MAX_VALUE, MAX_VALUE);

//             // argmin = _mm256_set_epi64(0,0,0,0);
//             uint64_t nearestNeighborsValue = MAX_VALUE;    
//             size_t argmin = 0;
//             seed = _mm256_set_epi64x(j+1, j+1, j+1, j+1);

//             if (pRawData->getSizeOfInstance(pInstance) > 4) {
//                 for (size_t i = 0; i < pRawData->getSizeOfInstance(pInstance) - 4; i+=4) {
               
//                     value = _mm256_set_epi64x(((pRawData->getNextElement(pInstance, i) + 1) * (j+1)), 
//                                         ((pRawData->getNextElement(pInstance, i+1) +1) * (j+1)),
//                                         ((pRawData->getNextElement(pInstance, i+2) +1) * (j+1)),
//                                         ((pRawData->getNextElement(pInstance, i+3) +1) * (j+1)));
//                     hashValue = mHash->hash_SSE_64(value, seed);
                    
//                     const uint64_t* minValue; 
//                     //add a pointer 
//                     minValue = (const uint64_t*) &hashValue;
//                     for (size_t k = 0; k < 4; k++) {
//                         if (nearestNeighborsValue <= minValue[k]){
//                             continue;
//                         } else {
//                             nearestNeighborsValue = minValue[k];
//                             argmin = pRawData->getNextElement(pInstance, i+k);
//                         }
//                     }
//                     // minimumVector = _mm256_min_epu64(hashValue, minimumVector);

//                     // compare all four hash values and store minimum for each element
//                     // argmin = _mm256_argmin_change_epi64(argmin, minimumVector, hashValue, value);

//                 }
//                 (*signature)[j] = argmin;
//             } else {
//                 uint64_t nearestNeighborsValue = MAX_VALUE;    
//                 size_t argmin_local = 0;
//                 size_t seed_local = 0;
//                 seed_local = j + 1;    
//                 for (size_t i = 0; i < pRawData->getSizeOfInstance(pInstance); i++) {
//                     uint64_t hashValue = mHash->hash_64((pRawData->getNextElement(pInstance, i) +1), seed_local, MAX_VALUE);
                
//                     if (hashValue < nearestNeighborsValue) {
//                         nearestNeighborsValue = hashValue;
//                         argmin_local = pRawData->getNextElement(pInstance, i);
//                     }
//                 }
//                 (*signature)[j] = argmin_local;
//             }
//     }
//     // reduce number of hash values by a factor of mShingleSize
//     if (mShingle) {

//         return shingle(signature);
//     }

//     return signature;
// }

// compute the signature for one instance
vsize_t* InverseIndex::computeSignature(SparseMatrixFloat* pRawData, const size_t pInstance) {

    // return computeSignatureAVX(pRawData, pInstance);
    return computeSignatureSSE(pRawData, pInstance);

    if (pRawData == NULL) return NULL;
    vsize_t* signature = new vsize_t(mNumberOfHashFunctions * mBlockSize);
    size_t argmin = 0;
    size_t seed = 0;
    for(size_t j = 0; j < mNumberOfHashFunctions * mBlockSize; ++j) {
            uint64_t nearestNeighborsValue = MAX_VALUE;    
            seed = j + 1;    
            for (size_t i = 0; i < pRawData->getSizeOfInstance(pInstance); i++) {
                uint64_t hashValue = mHash->hash((pRawData->getNextElement(pInstance, i) +1), seed, MAX_VALUE);
               
                if (hashValue < nearestNeighborsValue) {
                    nearestNeighborsValue = hashValue;
                    argmin = pRawData->getNextElement(pInstance, i);
                }
            }

            (*signature)[j] = argmin;
    }

    // reduce number of hash values by a factor of mShingleSize
    if (mShingle) {
        return shingle(signature);
    }

    return signature;
}

vsize_t* InverseIndex::shingle(vsize_t* pSignature) {
    
    if (pSignature == NULL) return NULL;
    vsize_t* signature = new vsize_t(mInverseIndexSize);
    size_t iterationSize = ceil((mNumberOfHashFunctions * mBlockSize) / mShingleSize);
    if (mShingle == 1) {
        
        // if 0 than combine hash values inside the block to one new hash value
        size_t signatureBlockValue;
        for (size_t i = 0; i < iterationSize; ++i) {
            signatureBlockValue = (*pSignature)[i*mShingleSize];
            
            for (size_t j = 1; j < mShingleSize; ++j) {
                signatureBlockValue = mHash->hash((*pSignature)[i*mShingleSize+j]+1, signatureBlockValue+1, MAX_VALUE);
            }
            (*signature)[i] = signatureBlockValue;
        }
        if (iterationSize != mInverseIndexSize) {
            signatureBlockValue = (*pSignature)[(iterationSize+1) * mShingleSize];
            for (size_t j = 0; j < mShingleSize && j + (iterationSize+1)*mShingleSize < pSignature->size(); ++j) {
                signatureBlockValue = mHash->hash((*pSignature)[(iterationSize+1)*mShingleSize + j]+1, signatureBlockValue+1, MAX_VALUE);
            }
            (*signature)[iterationSize] = signatureBlockValue;
        }
    }
    
    delete pSignature;
    return signature; 
}

vsize_t* InverseIndex::computeSignatureWTA(SparseMatrixFloat* pRawData, const size_t pInstance) {
    size_t sizeOfInstance = pRawData->getSizeOfInstance(pInstance);
    
    size_t mSeed = 42;
    size_t mK = mRangeK_Wta;
    
    vsize_t* signature = new vsize_t (mNumberOfHashFunctions * mBlockSize);
    if (sizeOfInstance < mK) {
        mK = sizeOfInstance;
    }
    KSizeSortedMap keyValue(mK);
    
    for (size_t i = 0; i < mNumberOfHashFunctions * mBlockSize; ++i) {
        
        for (size_t j = 0; j < sizeOfInstance; ++j) {
            size_t hashIndex = mHash->hash((pRawData->getNextElement(pInstance, j) +1), mSeed+i, MAX_VALUE);
            keyValue.insert(hashIndex, pRawData->getNextValue(pInstance, j));
        } 
        
        float maxValue = 0.0;
        size_t maxValueIndex = 0;
        
        for (size_t j = 0; j < mK; ++j) {
            if (keyValue.getValue(j) > maxValue) {
                maxValue = keyValue.getValue(j);
                maxValueIndex = keyValue.getKey(j);
                
            }
        }
        (*signature)[i] = maxValueIndex;//keyValue.getMaxValueIndex();
        keyValue.clear();
    }
    if (mShingle) {
        return shingle(signature);
    }
    return signature;
}

vvsize_t_p* InverseIndex::computeSignatureVectors(SparseMatrixFloat* pRawData, const bool pFitting) {
    
    if (mChunkSize <= 0) {
        mChunkSize = ceil(pRawData->size() / static_cast<float>(mNumberOfCores));
    }

    #ifdef OPENMP
    omp_set_dynamic(0);
    #endif

    vvsize_t_p* signatures = new vvsize_t_p(pRawData->size(), NULL);

    #ifdef CUDA
    if ((mCpuGpuLoadBalancing == 0 && mGpuHash == 0) || mHashAlgorithm == 1 ) {
    #endif
        #pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
        for (size_t instance = 0; instance < pRawData->size(); ++instance) {
            if (mHashAlgorithm == 0) {
                // use nearestNeighbors 

                (*signatures)[instance] = computeSignature(pRawData, instance);

            } else if (mHashAlgorithm == 1) {
                // use wta hash
                (*signatures)[instance] = computeSignatureWTA(pRawData, instance);
            }
        }
    #ifdef CUDA 
    } else {
        if (pFitting) {
            mInverseIndexCuda->computeSignaturesFittingOnGpu(pRawData, 0,
                                                    pRawData->size(),
                                                    pRawData->size(),
                                                    128, 128, 
                                                    mShingleSize,
                                                    mBlockSize,
                                                    signatures, mRangeK_Wta);
        } else { 
            mInverseIndexCuda->computeSignaturesQueryOnGpu(pRawData,  0,
                                                    pRawData->size(),
                                                    pRawData->size(),
                                                    128, 128, 
                                                    mShingleSize,
                                                    mBlockSize,
                                                    signatures, mRangeK_Wta);
            } 
    }
    #endif

    return signatures;
}
umap_uniqueElement* InverseIndex::computeSignatureMap(SparseMatrixFloat* pRawData) {
    mDoubleElementsQueryCount = 0;
    const size_t sizeOfInstances = pRawData->size();
    umap_uniqueElement* instanceSignature = new umap_uniqueElement();
    instanceSignature->reserve(sizeOfInstances);
    vvsize_t_p* signatures = computeSignatureVectors(pRawData, false);
    if (signatures != NULL) {
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
                    delete (*signatures)[i];
                }
            } 
        }
    }
    delete signatures;
    return instanceSignature;
}
void InverseIndex::fit(SparseMatrixFloat* pRawData, size_t pStartIndex) {

    vvsize_t_p* signatures = computeSignatureVectors(pRawData, true);
    
    if (signatures == NULL) return;
    // compute how often the inverse index should be pruned 
    size_t pruneEveryNInstances = ceil(signatures->size() * mPruneInverseIndexAfterInstance);
    #ifdef OPENMP
    omp_set_dynamic(0);
    #endif

    // store signatures in signatureStorage
    for (size_t i = 0; i < signatures->size(); ++i) {
        bool deleteSignature = false;
        if ((*signatures)[i] == NULL) continue;
        size_t signatureId = 0;
        for (size_t j = 0; j < pRawData->getSizeOfInstance(i); ++j) {
                signatureId = mHash->hash((pRawData->getNextElement(i, j) +1), (signatureId+1), MAX_VALUE);
        }

        auto itSignatureStorage = mSignatureStorage->find(signatureId);
        if (itSignatureStorage == mSignatureStorage->end()) {
            vsize_t* doubleInstanceVector = new vsize_t(1);
            (*doubleInstanceVector)[0] = i+pStartIndex;
            uniqueElement element;
            element.instances = doubleInstanceVector;
            element.signature = (*signatures)[i];
            mSignatureStorage->operator[](signatureId) = element;
        } else {
            {            
                mSignatureStorage->operator[](signatureId).instances->push_back(i+pStartIndex);
                mDoubleElementsStorageCount += 1;
                deleteSignature = true;
            }
        }     

        for (size_t j = 0; j < (*signatures)[i]->size(); ++j) {
            mInverseIndexStorage->insert(j, (*(*signatures)[i])[j], i+pStartIndex, mRemoveValueWithLeastSigificantBit);
        }

        // delete (*signatures)[i] if it appeares for the second time
        if (deleteSignature) {
             delete (*signatures)[i];
             deleteSignature = false;
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

    delete signatures;

}

neighborhood* InverseIndex::kneighbors(const umap_uniqueElement* pSignaturesMap, 
                                        const size_t pNneighborhood, 
                                        const bool pDoubleElementsStorageCount,
                                        const bool pNoneSingleInstance, float pRadius, bool pAbsoluteNumbers) {
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
    vvsize_t* neighbors = new vvsize_t();
    vvfloat* distances = new vvfloat();
    neighbors->resize(pSignaturesMap->size() + doubleElements);
    distances->resize(pSignaturesMap->size() + doubleElements);
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
        
        const vsize_t* signature = instanceId->second.signature; 
        if (signature == NULL) continue;
        
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
            vsize_t emptyVectorInt;
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
        vvsize_t neighborsForThisInstance(instanceId->second.instances->size());
        vvfloat distancesForThisInstance(instanceId->second.instances->size());

        for (size_t j = 0; j < neighborsForThisInstance.size(); ++j) {
            vsize_t neighborhoodVector;
            std::vector<float> distanceVector;
            for (auto it = neighborhoodVectorForSorting.begin(); it != neighborhoodVectorForSorting.end(); ++it) {
                
                float value = 0.0;
                if (pAbsoluteNumbers) {
                    value = (float) (*it).val;

                } else {
                    value = 1 - (((*it).val) / (float)(mMaximalNumberOfHashCollisions));
                }

                if (pRadius == -1.0) {
                    neighborhoodVector.push_back((*it).key);
                    distanceVector.push_back(value);
                } else {
                    if (value <= pRadius) {
                        neighborhoodVector.push_back((*it).key);
                        distanceVector.push_back(value);
                    } else {
                        break;
                    }
                }    
                
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