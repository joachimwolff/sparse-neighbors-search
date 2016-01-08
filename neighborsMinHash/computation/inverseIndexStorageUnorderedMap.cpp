#include "inverseIndexStorageUnorderedMap.h"
#ifdef OPENMP
#include <omp.h>
#endif
 
InverseIndexStorageUnorderedMap::InverseIndexStorageUnorderedMap(size_t pSizeOfInverseIndex, size_t pMaxBinSize) {
	mSignatureStorage = new vector__umapVector(pSizeOfInverseIndex);
	mMaxBinSize = pMaxBinSize;
	mKeys = new vvsize_t(pSizeOfInverseIndex, vsize_t());
	mValues = new vvsize_t(pSizeOfInverseIndex, vsize_t());
}
InverseIndexStorageUnorderedMap::~InverseIndexStorageUnorderedMap() {
	
}
size_t InverseIndexStorageUnorderedMap::size() const {
	return mSignatureStorage->size();
}
vsize_t* InverseIndexStorageUnorderedMap::getElement(size_t pVectorId, size_t pHashValue) {
	auto iterator = (*mSignatureStorage)[pVectorId].find(pHashValue);
	if (iterator != (*mSignatureStorage)[pVectorId].end()) {
		return &(iterator->second);
	}
	return NULL;
	
}
void InverseIndexStorageUnorderedMap::insert(size_t pVectorId, size_t pHashValue, size_t pInstance) {
#ifdef OPENMP
#pragma omp critical
#endif
    {	
        auto itHashValue_InstanceVector = (*mSignatureStorage)[pVectorId].find(pHashValue);
        // if for hash function h_i() the given hash values is already stored
        if (itHashValue_InstanceVector != (*mSignatureStorage)[pVectorId].end()) {
            // insert the instance id if not too many collisions (maxBinSize)
            if (itHashValue_InstanceVector->second.size() && itHashValue_InstanceVector->second.size() < mMaxBinSize) {
                // insert only if there wasn't any collisions in the past
                if (itHashValue_InstanceVector->second.size() > 0) {
                    itHashValue_InstanceVector->second.push_back(pInstance);
                }
            } else { 
                // too many collisions: delete stored ids. empty vector is interpreted as an error code 
                // for too many collisions
                itHashValue_InstanceVector->second.clear();
            }
        } else {
            // given hash value for the specific hash function was not avaible: insert new hash value
            vsize_t instanceIdVector;
            instanceIdVector.push_back(pInstance);
            (*mSignatureStorage)[pVectorId][pHashValue] = instanceIdVector;
        }
    }
}

std::map<size_t, size_t>* InverseIndexStorageUnorderedMap::getDistribution() {
    std::cout << __LINE__ << std::endl;
    std::map<size_t, size_t>* distribution = new std::map<size_t, size_t>();
    std::cout << __LINE__ << std::endl;
    
    for (auto it = mSignatureStorage->begin(); it != mSignatureStorage->end(); ++it) {
        for (auto itMap = it->begin(); itMap != it->end(); ++itMap) {
            (*distribution)[itMap->second.size()] += 1;
        }
    }
    std::cout << __LINE__ << std::endl;

    return distribution;
}
void InverseIndexStorageUnorderedMap::prune(int pValue) {
    for (auto it = mSignatureStorage->begin(); it != mSignatureStorage->end(); ++it) {
        vsize_t elementsToDelete;
        for (auto itMap = it->begin(); itMap != it->end(); ++itMap) {
            if (itMap->second.size() <= pValue) {
                elementsToDelete.push_back(itMap->first);
            }
            // (*distribution)[itMap->second.size()] += 1;
        }
        for (size_t i = 0; i < elementsToDelete.size(); ++i) {
            it->erase(elementsToDelete[i]);
        }
        elementsToDelete.clear();
    }
}

// if pRemoveHashFunctionWithLessEntriesAs == 0 remove every hash function 
// which has less entries than mean+standard deviation
// else: remove every hash function which has less entries than pRemoveHashFunctionWithLessEntriesAs
void InverseIndexStorageUnorderedMap::removeHashFunctionWithLessEntriesAs(int pRemoveHashFunctionWithLessEntriesAs) {
    if (pRemoveHashFunctionWithLessEntriesAs == 0) {
        int mean = 0;
        int variance = 0;
        for (size_t i = 0; i < mSignatureStorage->size(); ++i) {
            mean += (*mSignatureStorage)[i].size();
        }
        mean = mean / mSignatureStorage->size();
        for (size_t i = 0; i < mSignatureStorage->size(); ++i) {
            variance += pow(static_cast<int>((*mSignatureStorage)[i].size()) - mean, 2);
        }
        variance = variance / mSignatureStorage->size();
        int standardDeviation = sqrt(variance);
        for (size_t i = 0; i < mSignatureStorage->size(); ++i) {
            if ((*mSignatureStorage)[i].size() < mean + standardDeviation) {
                    (*mSignatureStorage)[i].clear();
            }
        }
    } else {
        for (size_t i = 0; i < mSignatureStorage->size(); ++i) {
            if ((*mSignatureStorage)[i].size() < pRemoveHashFunctionWithLessEntriesAs) {
                (*mSignatureStorage)[i].clear();
            }
        }
    }
}