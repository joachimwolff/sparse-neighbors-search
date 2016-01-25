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
	delete mSignatureStorage;
    delete mKeys;
    delete mValues;
}
size_t InverseIndexStorageUnorderedMap::size() const {
	return mSignatureStorage->size();
}
const vsize_t* InverseIndexStorageUnorderedMap::getElement(size_t pVectorId, size_t pHashValue) {
    // std::cout << __LINE__ << std::endl;
    
	auto iterator = (*mSignatureStorage)[pVectorId].find(pHashValue);
	if (iterator != (*mSignatureStorage)[pVectorId].end()) {
		return &(iterator->second);
	}
    // vsize_t foo;
	return NULL;
    // std::cout << __LINE__ << std::endl;
    
	
}
void InverseIndexStorageUnorderedMap::insert(size_t pVectorId, size_t pHashValue, size_t pInstance) {
    // std::cout << __LINE__ << std::endl;
    size_t insertValue = pHashValue | 0b11111111111111111111111111111100;
    if (insertValue == 0b11111111111111111111111111111100) {
        return;
    }
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

distributionInverseIndex* InverseIndexStorageUnorderedMap::getDistribution() {
    distributionInverseIndex* retVal = new distributionInverseIndex();
    std::map<size_t, size_t> distribution;
    vsize_t numberOfCreatedHashValuesPerHashFunction;
    vsize_t averageNumberOfValuesPerHashValue;
    vsize_t standardDeviationPerNumberOfValuesPerHashValue;
    size_t meanForNumberHashValues = 0;
    
    for (auto it = mSignatureStorage->begin(); it != mSignatureStorage->end(); ++it) {
        numberOfCreatedHashValuesPerHashFunction.push_back(it->size());
        meanForNumberHashValues += it->size();
        size_t mean = 0;
        
        for (auto itMap = it->begin(); itMap != it->end(); ++itMap) {
            distribution[itMap->second.size()] += 1;
            mean += itMap->second.size();
        }
        if (it->size() != 0 || mean != 0) {
            mean = mean / it->size();       
        }
        averageNumberOfValuesPerHashValue.push_back(mean);
        
        size_t variance = 0;
        for (auto itMap = it->begin(); itMap != it->end(); ++itMap) {
            variance += pow(static_cast<int>(itMap->second.size()) - mean, 2);
        }
        
        variance = variance / mSignatureStorage->size();
        int standardDeviation = sqrt(variance);
        standardDeviationPerNumberOfValuesPerHashValue.push_back(standardDeviation);
    }
    
    size_t varianceForNumberOfHashValues = 0;
    for (auto it = mSignatureStorage->begin(); it != mSignatureStorage->end(); ++it) {
       varianceForNumberOfHashValues += pow(it->size() - meanForNumberHashValues, 2);
    }
    
    retVal->mean = meanForNumberHashValues;
    retVal->standardDeviation = sqrt(varianceForNumberOfHashValues);
    
    retVal->totalCountForOccurenceOfHashValues = distribution;
    retVal->standardDeviationForNumberOfValuesPerHashValue = standardDeviationPerNumberOfValuesPerHashValue;
    retVal->meanForNumberOfValuesPerHashValue = averageNumberOfValuesPerHashValue;
    
    retVal->numberOfCreatedHashValuesPerHashFunction = numberOfCreatedHashValuesPerHashFunction;
    
    return retVal;
}
void InverseIndexStorageUnorderedMap::prune(int pValue) {
    // std::cout << __LINE__ << std::endl;
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
    // std::cout << __LINE__ << std::endl;
    
}

// if pRemoveHashFunctionWithLessEntriesAs == 0 remove every hash function 
// which has less entries than mean+standard deviation
// else: remove every hash function which has less entries than pRemoveHashFunctionWithLessEntriesAs
void InverseIndexStorageUnorderedMap::removeHashFunctionWithLessEntriesAs(int pRemoveHashFunctionWithLessEntriesAs) {
    // std::cout << __LINE__ << std::endl;
    
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
    // std::cout << __LINE__ << std::endl;
    
}