/**
 Copyright 2015 Joachim Wolff
 Master Thesis
 Tutors: Milad Miladi, Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/

#include <iostream>
#include <iterator>
#include <algorithm>
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

InverseIndex::InverseIndex(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions) {
    
    mNumberOfHashFunctions = pNumberOfHashFunctions;
    mBlockSize = pBlockSize;
    mNumberOfCores = pNumberOfCores;
    mChunkSize = pChunkSize;
    mMaxBinSize = pMaxBinSize;
    // mSizeOfNeighborhood = pSizeOfNeighborhood;
    mMinimalBlocksInCommon = pMinimalBlocksInCommon;
    mExcessFactor = pExcessFactor;
    mMaximalNumberOfHashCollisions = pMaximalNumberOfHashCollisions;
    mInverseIndexUmapVector = new std::vector<umapVector >();
    mSignatureStorage = new umap_uniqueElement();
    size_t inverseIndexSize = ceil(((float) mNumberOfHashFunctions / (float) mBlockSize)+1);
    mInverseIndexUmapVector->resize(inverseIndexSize);
}
 
InverseIndex::~InverseIndex() {
    delete mSignatureStorage;
    delete mInverseIndexUmapVector;
}
 // compute the signature for one instance
vsize_t InverseIndex::computeSignature(const vsize_t& featureVector) {

    vsize_t signatureHash;
    signatureHash.reserve(mNumberOfHashFunctions);

    for(size_t j = 0; j < mNumberOfHashFunctions; ++j) {
        size_t minHashValue = MAX_VALUE;
        for (size_t i = 0; i < featureVector.size(); ++i) {
            size_t hashValue = _size_tHashSimple((featureVector[i] +1) * (j+1) * A, MAX_VALUE);
            if (hashValue < minHashValue) {
                minHashValue = hashValue;
            }
        }
        signatureHash[j] = minHashValue;
    }
    // reduce number of hash values by a factor of blockSize
    size_t k = 0;
    vsize_t signature;
    signature.reserve((mNumberOfHashFunctions / mBlockSize) + 1);
    while (k < (mNumberOfHashFunctions)) {
        // use computed hash value as a seed for the next computation
        size_t signatureBlockValue = signatureHash[k];
        for (size_t j = 0; j < mBlockSize; ++j) {
            signatureBlockValue = _size_tHashSimple((signatureHash[k+j]) * signatureBlockValue * A, MAX_VALUE);
        }
        signature.push_back(signatureBlockValue);
        k += mBlockSize; 
    }
    return signature;
}
umap_uniqueElement* InverseIndex::computeSignatureMap(const umapVector* instanceFeatureVector) {
    std::cout << "start computing signature map..." << std::endl;
    mDoubleElementsQueryCount = 0;
    const size_t sizeOfInstances = instanceFeatureVector->size();
    umap_uniqueElement* instanceSignature = new umap_uniqueElement();
    (*instanceSignature).reserve(sizeOfInstances);
    if (mChunkSize <= 0) {
        mChunkSize = ceil(instanceFeatureVector->size() / static_cast<float>(mNumberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif


#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
    for(size_t index = 0; index < instanceFeatureVector->size(); ++index) {

        auto instanceId = instanceFeatureVector->begin();
        std::advance(instanceId, index);
        
        // compute unique id
        size_t signatureId = 0;
        for (auto itFeatures = instanceId->second.begin(); itFeatures != instanceId->second.end(); ++itFeatures) {
                signatureId = _size_tHashSimple((*itFeatures +1) * (signatureId+1) * A, MAX_VALUE);
        }

        // signature is in storage && 
        auto signatureIt = (*mSignatureStorage).find(signatureId);
        if (signatureIt != (*mSignatureStorage).end() && (instanceSignature->find(signatureId) != instanceSignature->end())) {
#pragma omp critical
            {
                instanceSignature->operator[](signatureId) = (*mSignatureStorage)[signatureId];
                instanceSignature->operator[](signatureId).instances.push_back(instanceId->first);
                mDoubleElementsQueryCount += (*mSignatureStorage)[signatureId].instances.size();
            }
            continue;
        }
        // for every hash function: compute the hash values of all features and take the minimum of these
        // as the hash value for one hash function --> h_j(x) = argmin (x_i of x) f_j(x_i)
        vsize_t signature = computeSignature(instanceId->second);
#pragma omp critical
        {
            if (instanceSignature->find(signatureId) == instanceSignature->end()) {
                vsize_t doubleInstanceVector(1);
                doubleInstanceVector[0] = instanceId->first;
                uniqueElement element;
                element.instances = doubleInstanceVector;
                element.signature = signature;
                instanceSignature->operator[](signatureId) = element;
            } else {
                instanceSignature->operator[](signatureId).instances.push_back(instanceId->first);
                mDoubleElementsQueryCount += 1;
            }
        }
    }
    std::cout << "Computing signature map done!" << std::endl;
    return instanceSignature;
}
void InverseIndex::fit(const umapVector* instanceFeatureVector) {
    std::cout << "start fitting" << std::endl;
    mDoubleElementsStorageCount = 0;
    size_t inverseIndexSize = ceil(((float) mNumberOfHashFunctions / (float) mBlockSize)+1);
    mInverseIndexUmapVector->resize(inverseIndexSize);
    std::cout << "Number of instances: " << instanceFeatureVector->size() << std::endl;
    if (mChunkSize <= 0) {
        mChunkSize = ceil(instanceFeatureVector->size() / static_cast<float>(mNumberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif

#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
    for(size_t index = 0; index < instanceFeatureVector->size(); ++index) {

        auto instanceId = instanceFeatureVector->begin();
        std::advance(instanceId, index);
        size_t signatureId = 0;

        for (auto itFeatures = instanceId->second.begin(); itFeatures != instanceId->second.end(); ++itFeatures) {
            signatureId = _size_tHashSimple((*itFeatures +1) * (signatureId+1) * A, MAX_VALUE);
        }
        vsize_t signature;

        auto itSignatureStorage = mSignatureStorage->find(signatureId);
        if (itSignatureStorage == mSignatureStorage->end()) {
            signature = computeSignature(instanceId->second);
        } else {
            signature = itSignatureStorage->second.signature;
        }


        // vmSize_tSize_t hashStorage;
        // vsize_t signatureHash(mNumberOfHashFunctions);
        // for every hash function: compute the hash values of all features and take the minimum of these
        // as the hash value for one hash function --> h_j(x) = argmin (x_i of x) f_j(x_i)
        


        // insert in inverse index
#pragma omp critical
        {    
            if (itSignatureStorage == mSignatureStorage->end()) {
                vsize_t doubleInstanceVector(1);
                doubleInstanceVector[0] = instanceId->first;
                uniqueElement element;
                element.instances = doubleInstanceVector;
                element.signature = signature;
                mSignatureStorage->operator[](signatureId) = element;

                // instanceSignature->operator[](signatureId) = (*mSignatureStorage)[signatureId];
                // instanceSignature->operator[](signatureId).instances.push_back(instanceId->first);
                // mDoubleElementsQueryCount += (*mSignatureStorage)[signatureId].first.size();
            } else {
                 mSignatureStorage->operator[](signatureId).instances.push_back(instanceId->first);
                 mDoubleElementsStorageCount += 1;
            }
            for (size_t j = 0; j < signature.size(); ++j) {
                auto itHashValue_InstanceVector = mInverseIndexUmapVector->operator[](j).find(signature[j]);
                // if for hash function h_i() the given hash values is already stored
                if (itHashValue_InstanceVector != mInverseIndexUmapVector->operator[](j).end()) {
                    // insert the instance id if not too many collisions (maxBinSize)
                    if (itHashValue_InstanceVector->second.size() < mMaxBinSize) {
                        // insert only if there wasn't any collisions in the past
                        if (itHashValue_InstanceVector->second.size() > 0) {
                            itHashValue_InstanceVector->second.push_back(instanceId->first);
                        }
                    } else { 
                        // too many collisions: delete stored ids. empty vector is interpreted as an error code 
                        // for too many collisions
                        itHashValue_InstanceVector->second.clear();
                    }
                } else {
                    // given hash value for the specific hash function was not avaible: insert new hash value
                    vsize_t instanceIdVector;
                    instanceIdVector.push_back(instanceId->first);
                    mInverseIndexUmapVector->operator[](j)[signature[j]] = instanceIdVector;
                }
            }
        }
    }
    // std::cout << "Fitting done" << std::endl;
    // for (auto it = mInverseIndexUmapVector->begin(); it != mInverseIndexUmapVector->end(); ++it) {
    //     for (auto it2 = it->begin(); it2 != it->end(); ++it2) {
    //         std::cout << "Unique signature: " << it2->first << " : ";

    //         for (auto it3 = it2->second.begin(); it3 != it2->second.end(); ++it3) {
    //             std::cout << *it3;
    //         }
    //     }
    //     std::cout << std::endl;
    // }
}

neighborhood InverseIndex::kneighbors(const umap_uniqueElement* pSignaturesMap, const int pNneighborhood, const bool pDoubleElementsStorageCount) {
    // std::cout << "Start computing neihgor" << std::endl;
    // std::cout << "235" << std::endl;

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
    // std::cout << "SIze of signauresMap: " << pSignaturesMap->size() << " doubleElements: " << doubleElements << std::endl;
    neighbors->resize(pSignaturesMap->size()+doubleElements);
    distances->resize(pSignaturesMap->size()+doubleElements);
    // std::cout << "Size of neighbors: " << neighbors->size() << std::endl;
    if (mChunkSize <= 0) {
        mChunkSize = ceil(mInverseIndexUmapVector->size() / static_cast<float>(mNumberOfCores));
    }
    // std::cout << "248" << std::endl;

#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
    for (size_t i = 0; i < pSignaturesMap->size(); ++i) {
        umap_uniqueElement::const_iterator instanceId = pSignaturesMap->begin();
        std::advance(instanceId, i); 
        
        std::unordered_map<size_t, size_t> neighborhood;
        const vsize_t signature = instanceId->second.signature;
        for (size_t j = 0; j < signature.size(); ++j) {
            size_t hashID = signature[j];
            if (hashID != 0 && hashID != MAX_VALUE) {
                size_t collisionSize = 0;
                umapVector::const_iterator instances = mInverseIndexUmapVector->at(j).find(hashID);
                if (instances != mInverseIndexUmapVector->at(j).end()) {
                    collisionSize = instances->second.size();
                } else { 
                    continue;
                }

                if (collisionSize < mMaxBinSize && collisionSize > 0) {
                    for (size_t k = 0; k < instances->second.size(); ++k) {
                        neighborhood[instances->second.at(k)] += 1;
                    }
                }
            }
        }
    // std::cout << "275" << std::endl;

        std::vector< sort_map > neighborhoodVectorForSorting;
        
        for (auto it = neighborhood.begin(); it != neighborhood.end(); ++it) {
            sort_map mapForSorting;
            mapForSorting.key = (*it).first;
            mapForSorting.val = (*it).second;
            neighborhoodVectorForSorting.push_back(mapForSorting);
        }
        std::sort(neighborhoodVectorForSorting.begin(), neighborhoodVectorForSorting.end(), mapSortDescByValue);
        vint neighborhoodVector;
        std::vector<float> distanceVector;

        size_t sizeOfNeighborhoodAdjusted = std::min(static_cast<size_t>(pNneighborhood * mExcessFactor), neighborhoodVectorForSorting.size());
        size_t count = 0;

        for (auto it = neighborhoodVectorForSorting.begin();
                it != neighborhoodVectorForSorting.end(); ++it) {
            neighborhoodVector.push_back((*it).key);
            distanceVector.push_back(1 - ((*it).val / static_cast<float>(mMaximalNumberOfHashCollisions)));
            ++count;
            if (count == sizeOfNeighborhoodAdjusted) {
                break;
            }
        }
    // std::cout << "301" << std::endl;

        // guarantee that at least k-neighbors + 1 elements are in each vector.
        // +1 : in the case of nearest neighbors query for all instances, 
        //      the first value will be cut in the python interface
        // int appendSize = 0;
        // appendSize = pNneighborhood - neighborhoodVector.size();
        // for (int j = 0; j < appendSize+1; ++j) {
        //     neighborhoodVector.push_back(-1);
        //     distanceVector.push_back(0);
        // }
        
#pragma omp critical
        { 
            // std::cout << "instances with nearest neighbors: ";
            for (size_t j = 0; j < instanceId->second.instances.size(); ++j) {
                // std::cout << instanceId->second.instances[j] << std::endl;
                (*neighbors)[instanceId->second.instances[j]] = neighborhoodVector;
                (*distances)[instanceId->second.instances[j]] = distanceVector;
            }
            // std::cout << std::endl;
        }
    }
    // std::cout << "326" << std::endl;

    neighborhood neighborhood_;
    neighborhood_.neighbors = neighbors;
    neighborhood_.distances = distances;
    // std::cout << "End computing neighbot in inverseIndex." << std::endl;
    // std::cout << "[ ";
    // for (auto it = neighborhood_.neighbors->begin(); it != neighborhood_.neighbors->end(); ++it) {
    //     std::cout << "[ ";
    //     for (auto it2 = it->begin(); it2 != it->end(); ++it2) {
    //         std::cout << (*it2) << " ";
    //     }
    //     std::cout << " ]" << std::endl;
    // }
    // std::cout << " ]" << std::endl;


    // std::cout << "336" << std::endl;

    return neighborhood_;
}