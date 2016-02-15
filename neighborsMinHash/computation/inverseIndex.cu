/**
 Copyright 2016 Joachim Wolff
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
#include "kernel.h"

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
        // signature is in storage && 
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
    std::cout << __LINE__ << std::endl;
    size_t pruneEveryNIterations = pRawData->size() * mPruneInverseIndexAfterInstance;
    size_t pruneCount = 0;
    mDoubleElementsStorageCount = 0;
    if (mChunkSize <= 0) { 
        mChunkSize = ceil(pRawData->size() / static_cast<float>(mNumberOfCores));
    }
    std::cout << __LINE__ << std::endl;
    
#ifdef OPENMP
    
#endif
#ifndef OPENMP
    // mNumberOfCores = 1;
#endif
    vvsize_t_p signatures;
    omp_set_nested(1);
    omp_set_dynamic(0);
    // omp_set_num_threads(mNumberOfCores);
    // omp_set_nested(1);
    
#pragma omp parallel num_threads(mNumberOfCores)
    {
        vvsize_t_p signaturesPerThread;
        // (pRawData->size() / mNumberOfCores);
        // size_t substractFactor = omp_get_thread_num() - 1 * pRawData->size() / 2 / mNumberOfCores;
        #pragma omp master nowait
        {
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
            
            // check if enough memory is available on the gpu 
            size_t memory_total = 0;
            size_t memory_free = 0;
            size_t iterations = 1;
            cudaMemGetInfo(&memory_free, &memory_total);
            std::cout << "memory total: " << memory_total << " memory free: " << memory_free << std::endl;
            std::cout << "sizeof)size_t) : " << sizeof(size_t) << std::endl;
            std::cout << "Needed memory: " << pRawData->getNumberOfInstances()  * mNumberOfHashFunctions * sizeof(size_t) << std::endl;
            if (memory_free >= pRawData->getNumberOfInstances()  * mNumberOfHashFunctions * sizeof(size_t)) {
                iterations = ceil(pRawData->getNumberOfInstances()  * mNumberOfHashFunctions * sizeof(size_t) / static_cast<float>(memory_free));
            }
            std::cout << "Iterations: " << iterations << std::endl;
            size_t start = 0;
            size_t end = pRawData->getNumberOfInstances() / iterations;
            size_t windowSize = pRawData->getNumberOfInstances() / iterations;
            size_t* instancesHashValues = (size_t*) malloc(pRawData->getNumberOfInstances() / iterations * mNumberOfHashFunctions * sizeof(size_t));
            std::cout << __LINE__ << std::endl;
            
            // memory for the inverse index on the gpu.
            // for each instance the number of hash functions
            cudaMalloc((void **) &mDev_ComputedSignaturesPerInstance,
                    pRawData->getNumberOfInstances() / iterations  * mNumberOfHashFunctions * sizeof(size_t));
            
            for (size_t i = 0; i < iterations; ++i) {
                
                fitCuda<<<64, 128, mNumberOfHashFunctions * sizeof(size_t)>>>
                (mDev_FeatureList, 
                mDev_SizeOfInstanceList,  
                mNumberOfHashFunctions, 
                pRawData->getMaxNnz(),
                        mDev_ComputedSignaturesPerInstance, 
                        end, start);
                std::cout << __LINE__ << std::endl;
                        
                cudaMemcpy(instancesHashValues, mDev_ComputedSignaturesPerInstance, 
                            pRawData->getNumberOfInstances()/iterations * mNumberOfHashFunctions * sizeof(size_t),
                            cudaMemcpyDeviceToHost);
                std::cout << __LINE__ << std::endl;
                            
                for(size_t i = 0; i < pRawData->getNumberOfInstances() / iterations; ++i) {
                    // printf("Instance: %zu of %zu: ", i, pRawData->getNumberOfInstances());
                    vsize_t* instance = new vsize_t(mNumberOfHashFunctions);
                    for (size_t j = 0; j < mNumberOfHashFunctions; ++j) {
                        (*instance)[j] = instancesHashValues[i*mNumberOfHashFunctions + j];
                        // printf("%zu,", instancesHashValues[i*mNumberOfHashFunctions + j]);
                    }
                    signaturesPerThread.push_back(instance);
                    // printf("\n");
                }
                
                #pragma omp critical
                signatures.insert(signatures.end(), signaturesPerThread.begin(), signaturesPerThread.end());
                signaturesPerThread.clear();
                start = end+1;
                end = end + windowSize;
            }
            cudaFree(mDev_ComputedSignaturesPerInstance);
            
        }
       
        // std::cout << __LINE__ << std::endl;
        // mInverseIndexStorage->reserveSpaceForMaps(pRawData->size() / 2);
        // for (size_t i = 0; i < signatures.size(); ++i) {
        //     // std::cout << "i: " << i << std::endl;
        //     for (size_t j = 0; j < signatures[i]->size(); ++j) {
        //         // std::cout << signatures[i]->operator[](j) << ",";
                
        //         mInverseIndexStorage->insert(j, signatures[i]->operator[](j), i, mRemoveValueWithLeastSigificantBit);
        //     }
        //     // std::cout << std::endl;
        // }
    #pragma omp for nowait schedule(static)
            for (size_t instance = pRawData->size() / 2; instance < pRawData->size(); ++instance) {
                std::cout << "for no wait thread id: " << omp_get_thread_num() << ", ";
                // if (omp_get_thread_num() == 0) break;
                if (mHashAlgorithm == 0) {
                    // use minHash
                    signaturesPerThread.push_back(computeSignature(pRawData, instance));
                    
                } else if (mHashAlgorithm == 1) {
                    // use wta hash
                    signaturesPerThread.push_back(computeSignatureWTA(pRawData, instance));
                }
            }
                
      #pragma omp for schedule(static) ordered
            for(int i=0; i < omp_get_num_threads(); i++) {
            #pragma omp ordered
                signatures.insert(signatures.end(), signaturesPerThread.begin(), signaturesPerThread.end());
            }       
        
    } 
// #ifndef OPENMP
//     signatures = signaturesPerThread;
// #endif

    // for (size_t i = 0; i < signatures.size(); ++i) {
    //     std::cout << "Instance " << i << " foo size: "<< signatures[i]->size() << std::endl;
    //     for (size_t j = 0; j < signatures[i]->size(); ++j) {
    //         std::cout << (signatures[i])->operator[](j) << ", ";
    //     }
    //     std::cout << std::endl;
    // }
    
    // std::cout << "Size of signatue: " << signatures.size() << std::endl;

    // std::cout << __LINE__ << std::endl;
    // return;
    // add gpu part
    // merge gpu signatures with cpu signatures
    
//     std::cout << "insert to inverse index: " << std::endl;
// // create inverse index 
// #ifdef OPENMP
//     omp_set_dynamic(0);
// #endif
// #ifdef OPENMP
// #pragma omp parallel num_threads(1)
// #endif
//     { 
//     // std::cout << __LINE__ << std::endl;
        
//          vector__umapVector_ptr inverseIndex (mInverseIndexSize);
//         //  size_t substractFactor = omp_get_thread_num() * (mInverseIndexSize / mNumberOfCores);
         
//          for (size_t i = 0; i < mInverseIndexSize; ++i) {
//              inverseIndex[i] = new umapVector_ptr();
//              inverseIndex[i]->reserve(pRawData->size() / 2);
//          }
//     // std::cout << __LINE__ << std::endl;
         
         
// #ifdef OPENMP
// #pragma omp for  
// #endif

//         for (size_t i = 0; i < mInverseIndexSize; ++i) {
//             for (size_t j = 0; j < signatures.size(); ++j) {
//                 size_t hashValue = (signatures[j])->operator[](i);
//                 if (mRemoveValueWithLeastSigificantBit) {
//                     size_t leastSignificantBits = 0b11111111111111111111111111111111 << mRemoveValueWithLeastSigificantBit;
//                     size_t insertValue = hashValue | leastSignificantBits;
//                     if (insertValue == leastSignificantBits) {
//                         continue;
//                     }
//                 }       
//                 auto itHashValue_InstanceVector = inverseIndex[i]->find(hashValue);

//                 // if for hash function h_i() the given hash values is already stored
//                 if (itHashValue_InstanceVector != inverseIndex[i]->end()) {
//                     // insert the instance id if not too many collisions (maxBinSize)
//                     if (itHashValue_InstanceVector->second->size() && itHashValue_InstanceVector->second->size() < mMaxBinSize) {
//                         // insert only if there wasn't any collisions in the past
//                         if (itHashValue_InstanceVector->second->size() > 0) {
//                             itHashValue_InstanceVector->second->push_back(j);
//                         }
//                     } else { 
//                         // too many collisions: delete stored ids. empty vector is interpreted as an error code 
//                         // for too many collisions
//                         itHashValue_InstanceVector->second->clear();
//                     }
//                 } else {
//                     // given hash value for the specific hash function was not avaible: insert new hash value
//                     vsize_t* instanceIdVector = new vsize_t(1);
//                     (*instanceIdVector)[0] = j;
//                     inverseIndex[i]->operator[](hashValue) = instanceIdVector;
//                 }       
//             }
//         }

// #ifdef OPENMP
// #pragma omp for schedule(static) ordered
//         for(int i=0; i<omp_get_num_threads(); i++) {
//             #pragma omp ordered
//             mInverseIndexStorage->insert(inverseIndex.begin(), inverseIndex.end());
//         }       
// #endif
//     } 
    
    // for (size_t i = 0; i < mInverseIndexStorage->size(); ++i) {
    //     std::cout << "hash function: " << i << std::endl;
    //     for (auto it = mInverseIndexStorage->getIndex()->operator[](i)->begin();
    //             it != mInverseIndexStorage->getIndex()->operator[](i)->end(); ++it) {
    //                 std::cout << "hashValue: " << it->first << ": ";
    //                 for (auto itVec = it->second->begin(); itVec != it->second->end(); ++itVec) {
    //                     std::cout << *itVec << ",";
    //                 }
    //                 std::cout << std::endl;
    //             }
    //                 std::cout << std::endl;
                
    // }
    
// #ifdef OPENMP
//     omp_set_dynamic(0);
// #endif
// #ifdef OPENMP
// #pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
// #endif
//     for (size_t index = 0; index < pRawData->size(); ++index) {
//         size_t signatureId = 0;
//         for (size_t j = 0; j < pRawData->getSizeOfInstance(index); ++j) {
//             signatureId = mHash->hash((pRawData->getNextElement(index, j) +1), (signatureId+1), MAX_VALUE);
//         }
//         vsize_t* signature;
//         auto itSignatureStorage = mSignatureStorage->find(signatureId);
//         if (itSignatureStorage == mSignatureStorage->end()) {
//             if (mHashAlgorithm == 0) {
//                 // use minHash
//                 signature = computeSignature(pRawData, index);
//             } else if (mHashAlgorithm == 1) {
//                 // use wta hash
//                 signature = computeSignatureWTA(pRawData, index);
//             }
//         } else {
//             signature = itSignatureStorage->second.signature;
//         }
// #ifdef OPENMP
// #pragma omp critical
// #endif
//         {   
//             ++pruneCount;
//             if (itSignatureStorage == mSignatureStorage->end()) {
//                 vsize_t* doubleInstanceVector = new vsize_t(1);
//                 (*doubleInstanceVector)[0] = index;
//                 uniqueElement element;
//                 element.instances = doubleInstanceVector;
//                 element.signature = signature;
//                 mSignatureStorage->operator[](signatureId) = element;
//             } else {
//                  mSignatureStorage->operator[](signatureId).instances->push_back(index);
//                  mDoubleElementsStorageCount += 1;
//             }
//         }
        
//         for (size_t j = 0; j < signature->size(); ++j) {
//             mInverseIndexStorage->insert(j, (*signature)[j], index, mRemoveValueWithLeastSigificantBit);
//         }
        
//         if (mPruneInverseIndexAfterInstance > 0) {
// #ifdef OPENMP
// #pragma omp critical
// #endif
//             {
//                 if (pruneCount >= pruneEveryNIterations) {
//                     pruneCount = 0;
                    
//                     if (mPruneInverseIndex > 0) {
//                         mInverseIndexStorage->prune(static_cast<size_t>(mPruneInverseIndex));
//                     }
//                     if (mRemoveHashFunctionWithLessEntriesAs >= 0) {
//                         mInverseIndexStorage->removeHashFunctionWithLessEntriesAs(static_cast<size_t>(mRemoveHashFunctionWithLessEntriesAs));
//                     }
//                 }
//             }           
//         }
//     }
//     	std::cout << __LINE__ << std::endl;
    
//     if (mPruneInverseIndex > 0) {
//         mInverseIndexStorage->prune(mPruneInverseIndex);
//     }
//     	std::cout << __LINE__ << std::endl;
    
//     if (mRemoveHashFunctionWithLessEntriesAs >= 0) {
//         mInverseIndexStorage->removeHashFunctionWithLessEntriesAs(static_cast<size_t>(mRemoveHashFunctionWithLessEntriesAs));
//     }
//     // for (std::cout << )
//     	std::cout << __LINE__ << std::endl;
//         std::cout << "Number of hash function: " << mInverseIndexStorage->size() << std::endl;
//         for (size_t i = 0; i < mInverseIndexStorage->size(); ++i) {
//             std::cout << "hash function: " << i << " Size: " << mInverseIndexStorage->getIndex()->operator[](i)->size();
//             std::cout << " Load factor: " << mInverseIndexStorage->getIndex()->operator[](i)->load_factor() << std::endl;
            
//         }
    
}

neighborhood* InverseIndex::kneighbors(const umap_uniqueElement* pSignaturesMap, 
                                        const size_t pNneighborhood, const bool pDoubleElementsStorageCount) {
                                            // std::cout << "kneighbors inverseIndex" << std::endl;

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
        neighborhood.reserve(mMaxBinSize*2);
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
        size_t numberOfElementsToSort = pNneighborhood;
        if (pNneighborhood > neighborhoodVectorForSorting.size()) {
            numberOfElementsToSort = neighborhoodVectorForSorting.size();
        }
        
        std::partial_sort(neighborhoodVectorForSorting.begin(), 
                            neighborhoodVectorForSorting.begin()+numberOfElementsToSort, 
                            neighborhoodVectorForSorting.end(), mapSortDescByValue);
        size_t sizeOfNeighborhoodAdjusted;
        if (pNneighborhood == MAX_VALUE) {
            sizeOfNeighborhoodAdjusted = std::min(static_cast<size_t>(pNneighborhood), neighborhoodVectorForSorting.size());
        } else {
            sizeOfNeighborhoodAdjusted = std::min(static_cast<size_t>(pNneighborhood * mExcessFactor), neighborhoodVectorForSorting.size());
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