#include "inverseIndexStorageUnorderedMap.h"
#ifdef OPENMP
#include <omp.h>
#endif
 
InverseIndexStorageUnorderedMap::InverseIndexStorageUnorderedMap(size_t pSizeOfInverseIndex, size_t pMaxBinSize) {
	mInverseIndex = new vector__umapVector(pSizeOfInverseIndex);
	mMaxBinSize = pMaxBinSize;
}
InverseIndexStorageUnorderedMap::~InverseIndexStorageUnorderedMap() {
	delete mInverseIndex;
}
size_t InverseIndexStorageUnorderedMap::size() const {
	return mInverseIndex->size();
}
const vsize_t* InverseIndexStorageUnorderedMap::getElement(size_t pVectorId, size_t pHashValue) {
    if (pVectorId < mInverseIndex->size()) {
        auto iterator = (*mInverseIndex)[pVectorId].find(pHashValue);
        if (iterator != (*mInverseIndex)[pVectorId].end()) {
            return &(iterator->second);
        }
    }
	return NULL; 
}
void InverseIndexStorageUnorderedMap::insert(size_t pVectorId, size_t pHashValue, size_t pInstance, 
                        size_t pRemoveValueWithLeastSigificantBit) {
    if (pVectorId >= mInverseIndex->size()) return;
    if (pRemoveValueWithLeastSigificantBit) {
        size_t leastSignificantBits = 0b11111111111111111111111111111111 << pRemoveValueWithLeastSigificantBit;
        size_t insertValue = pHashValue | leastSignificantBits;
        if (insertValue == leastSignificantBits) {
            return;
        }
    }
    
#ifdef OPENMP
#pragma omp critical
#endif
    {	
        auto itHashValue_InstanceVector = (*mInverseIndex)[pVectorId].find(pHashValue);

        // if for hash function h_i() the given hash values is already stored
        if (itHashValue_InstanceVector != (*mInverseIndex)[pVectorId].end()) {
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
            (*mInverseIndex)[pVectorId][pHashValue] = instanceIdVector;
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
    
    for (auto it = mInverseIndex->begin(); it != mInverseIndex->end(); ++it) {
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
        
        variance = variance / mInverseIndex->size();
        int standardDeviation = sqrt(variance);
        standardDeviationPerNumberOfValuesPerHashValue.push_back(standardDeviation);
    }
    
    size_t varianceForNumberOfHashValues = 0;
    for (auto it = mInverseIndex->begin(); it != mInverseIndex->end(); ++it) {
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
void InverseIndexStorageUnorderedMap::prune(size_t pValue) { 
    for (auto it = mInverseIndex->begin(); it != mInverseIndex->end(); ++it) {
        vsize_t elementsToDelete;
        for (auto itMap = it->begin(); itMap != it->end(); ++itMap) {
            if (itMap->second.size() <= pValue) {
                elementsToDelete.push_back(itMap->first);
            }
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
void InverseIndexStorageUnorderedMap::removeHashFunctionWithLessEntriesAs(size_t pRemoveHashFunctionWithLessEntriesAs) {
    // std::cout << __LINE__ << std::endl;
    
    if (pRemoveHashFunctionWithLessEntriesAs == 0) {
        size_t mean = 0;
        size_t variance = 0;
        for (size_t i = 0; i < mInverseIndex->size(); ++i) {
            mean += (*mInverseIndex)[i].size();
        }
        mean = mean / mInverseIndex->size();
        for (size_t i = 0; i < mInverseIndex->size(); ++i) {
            variance += pow(static_cast<int>((*mInverseIndex)[i].size()) - mean, 2);
        }
        variance = variance / mInverseIndex->size();
        size_t standardDeviation = sqrt(variance);
        for (size_t i = 0; i < mInverseIndex->size(); ++i) {
            if ((*mInverseIndex)[i].size() < mean + standardDeviation) {
                    (*mInverseIndex)[i].clear();
            }
        }
    } else {
        for (size_t i = 0; i < mInverseIndex->size(); ++i) {
            if ((*mInverseIndex)[i].size() < pRemoveHashFunctionWithLessEntriesAs) {
                (*mInverseIndex)[i].clear();
            }
        }
    }
}