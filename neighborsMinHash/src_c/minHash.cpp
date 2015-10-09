/**
 Copyright 2015 Joachim Wolff
 Master Project
 Tutors: Milad Miladi, Fabrizio Costa
 Summer semester 2015

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/

#include <minHash.h>

MinHash::MinHash(size_t numberOfHashFunctions, size_t blockSize,
                    size_t numberOfCores, size_t chunkSize,
                    size_t maxBinSize, size_t lazyFitting,
                    size_t sizeOfNeighborhood, size_t minimalBlocksInCommon,
                    size_t excessFactor, float maximalNumberOfHashCollisions) {
    
    numberOfHashFunctions = numberOfHashFunctions;
    blockSize = blockSize;
    numberOfCores = numberOfCores;
    chunkSize = chunkSize;
    maxBinSize = maxBinSize;
    lazyFitting = lazyFitting;
    sizeOfNeighborhood = sizeOfNeighborhood;
    minimalBlocksInCommon = minimalBlocksInCommon;
    excessFactor = excessFactor;
    maximalNumberOfHashCollisions = maximalNumberOfHashCollisions;
    signatureStorage = NULL;
    inverseIndex = NULL;
}

MinHash::~MinHash() {
    delete signatureStorage;
    delete inverseIndex;
}
 // compute the signature for one instance
vsize_t MinHash::computeSignature(const vsize_t& instanceFeatureVector) {
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
#pragma omp parallel for schedule(static, chunkSize) num_threads(numberOfCores)
    {
        for(size_t j = 0; j < numberOfHashFunctions; ++j) {
            size_t minHashValue = MAX_VALUE;
            for (auto itFeatures = instanceId->second.begin(); itFeatures != instanceId->second.end(); ++itFeatures) {
                size_t hashValue = _size_tHashSimple((*itFeatures +1) * (j+1) * A, MAX_VALUE);
                if (hashValue < minHashValue) {
                    minHashValue = hashValue;
                }
            }
            signatureHash[j] = minHashValue;
        }
    } 
    // reduce number of hash values by a factor of blockSize
    size_t k = 0;
    vsize_t signature;
    signature.reserve((numberOfHashFunctions / blockSize) + 1);
    while (k < (numberOfHashFunctions)) {
        // use computed hash value as a seed for the next computation
        size_t signatureBlockValue = signatureHash[k];
        for (size_t j = 0; j < blockSize; ++j) {
            signatureBlockValue = _size_tHashSimple((signatureHash[k+j]) * signatureBlockValue * A, MAX_VALUE);
        }
        signature.push_back(signatureBlockValue);
        k += blockSize; 
    }
    return signature;
}
vvsize_t MinHash::computeSignature_2(const umapVector& instanceFeatureVector) {

    const size_t sizeOfInstances = instanceFeatureVector.size();
    vvsize_t instanceSignature;
    instanceSignature.resize(sizeOfInstances);
    if (chunkSize <= 0) {
        chunkSize = ceil(instanceFeatureVector.size() / static_cast<float>(numberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
#pragma omp parallel for schedule(static, chunkSize) num_threads(numberOfCores)
    for(size_t index = 0; index < instanceFeatureVector.size(); ++index) {

        auto instanceId = instanceFeatureVector.begin();
        std::advance(instanceId, index);
        // compute unique id
        size_t signatureId = 0;
        for (auto itFeatures = instanceId->second.begin(); itFeatures != instanceId->second.end(); ++itFeatures) {
                signatureId = _size_tHashSimple((*itFeatures +1) * (signatureId+1) * A, MAX_VALUE);
        }
        auto signatureIt = signatureStorage.find(signatureId);
        if (signatureIt != signatureStorage.end()) {
#pragma critical
            instanceSignature[index] = signatureIt->second;
            continue;
        }
        // for every hash function: compute the hash values of all features and take the minimum of these
        // as the hash value for one hash function --> h_j(x) = argmin (x_i of x) f_j(x_i)
        vsize_t signature = computeSignature(instanceId->second);
#pragma critical

        instanceSignature[index] = signature;
        signatureStorage[signatureId] = signature;
    }
    return instanceSignature;
}
void MinHash::computeInverseIndex(umapVector& instanceFeatureVector) {

    // if NULL than the inverse index is new created. Otherwise it is extended.
    if (inverseIndex == NULL) {
        inverseIndex = new std::vector<umapVector >();
    }
    if (signatureStorage == NULL) {
        signatureStorage = new umapVector();
    }
    size_t inverseIndexSize = ceil(((float) numberOfHashFunctions / (float) blockSize)+1);
    inverseIndex->resize(inverseIndexSize);
    if (chunkSize <= 0) {
        chunkSize = ceil(instance_featureVector.size() / static_cast<float>(numberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
#pragma omp parallel for schedule(static, chunkSize) num_threads(numberOfCores)
    for(size_t index = 0; index < instance_featureVector.size(); ++index){

        auto instanceId = instance_featureVector.begin();
        std::advance(instanceId, index);

        vmSize_tSize_t hashStorage;
        vsize_t signatureHash(numberOfHashFunctions);
        // for every hash function: compute the hash values of all features and take the minimum of these
        // as the hash value for one hash function --> h_j(x) = argmin (x_i of x) f_j(x_i)
        vsize_t signature = computeSignature(instanceId->second);

        if (!lazyFitting) {
            for (auto itFeatures = instanceId->second.begin(); itFeatures != instanceId->second.end(); ++itFeatures) {
                signatureId = _size_tHashSimple((*itFeatures +1) * (signatureId+1) * A, MAX_VALUE);
            }
            signatureStorage[signatureId] = signature;
        } else {
            signatureStorage[index] = signature;
        }
        // insert in inverse index
#pragma omp critical
        for (size_t j = 0; j < signature.size(); ++j) {
            auto itHashValue_InstanceVector = inverseIndex[j].find(signature[j]);
            // if for hash function h_i() the given hash values is already stored
            if (itHashValue_InstanceVector != inverseIndex[j].end()) {
                // insert the instance id if not too many collisions (maxBinSize)
                if (itHashValue_InstanceVector->second.size() < maxBinSize) {
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
                inverseIndex[j][signature[j]] = instanceIdVector;
            }
        }
    }
}

std::pair<vvsize_t , vvfloat > MinHash::computeNeighbors(const vvsize_t signatures) {
#ifdef OPENMP
    omp_set_dynamic(0);
#endif

    std::pair<vvsize_t , vvfloat > returnVector;

    vvsize_t neighbors;
    vvfloat distances;

    neighbors.resize(signatures.size());
    distances.resize(signatures.size());
    if (chunkSize <= 0) {
        chunkSize = ceil(inverseIndex.size() / static_cast<float>(numberOfCores));
    }
#pragma omp parallel for schedule(static, chunkSize) num_threads(numberOfCores)
    for (size_t i = 0; i < signatures.size(); ++i) {
        std::map<size_t, size_t> neighborhood;
        for (size_t j = 0; j < signatures[i].size(); ++j) {
            size_t hashID = signatures[i][j];
            if (hashID != 0 && hashID != MAX_VALUE) {
                size_t collisionSize = 0;
                auto instances = inverseIndex.at(j).find(hashID);
                if (instances != inverseIndex.at(j).end()) {
                    collisionSize = instances->second.size();
                } else {
                    continue;
                }
                if (collisionSize < maxBinSize && collisionSize > 0) {
                    for (size_t k = 0; k < instances->second.size(); ++k) {
                        neighborhood[instances->second.at(k)] += 1;
                    }
                }
            }
        }
        std::vector< sort_map > neighborhoodVectorForSorting;

        sort_map mapForSorting;
        for (auto it = neighborhood.begin(); it != neighborhood.end(); ++it) {
            mapForSorting.key = (*it).first;
            mapForSorting.val = (*it).second;
            neighborhoodVectorForSorting.push_back(mapForSorting);
        }
        std::sort(neighborhoodVectorForSorting.begin(), neighborhoodVectorForSorting.end(), mapSortDescByValue);
        vsize_t neighborhoodVector;
        std::vector<float> distanceVector;

        size_t sizeOfNeighborhoodAdjusted = std::min(static_cast<size_t>(sizeOfNeighborhood * excessFactor), neighborhoodVectorForSorting.size());
        size_t count = 0;

        for (auto it = neighborhoodVectorForSorting.begin();
                it != neighborhoodVectorForSorting.end(); ++it) {
            neighborhoodVector.push_back((*it).key);
            distanceVector.push_back(1 - ((*it).val / maximalNumberOfHashCollisions));
            ++count;
            if (count == sizeOfNeighborhoodAdjusted) {
                break;
            }
        }
        neighbors[i] = neighborhoodVector;
        distances[i] = distanceVector;
    }

    returnVector.first = neighbors;
    returnVector.second = distances;
    return returnVector;
}
