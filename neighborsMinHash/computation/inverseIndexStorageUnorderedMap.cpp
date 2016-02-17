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

#include "inverseIndexStorageUnorderedMap.h"

InverseIndexStorageUnorderedMap::InverseIndexStorageUnorderedMap(size_t pSizeOfInverseIndex, size_t pMaxBinSize) {
    mInverseIndex = new vector__umapVector_ptr(pSizeOfInverseIndex);
     for (size_t i = 0; i < mInverseIndex->size(); ++i) {
       
        mInverseIndex->operator[](i) = new umapVector_ptr();
    }
	mMaxBinSize = pMaxBinSize;
}
InverseIndexStorageUnorderedMap::~InverseIndexStorageUnorderedMap() {
    for (size_t i = 0; i < mInverseIndex->size(); ++i) {
        if (mInverseIndex->operator[](i) == NULL) continue;        
        for (auto it = mInverseIndex->operator[](i)->begin(); it != mInverseIndex->operator[](i)->end(); ++it) {
            if (it->second == NULL) continue;
            delete it->second;
        }
        delete mInverseIndex->operator[](i);
    }
	delete mInverseIndex;
}
size_t InverseIndexStorageUnorderedMap::size() const {
	return mInverseIndex->size();
}
void InverseIndexStorageUnorderedMap::reserveSpaceForMaps(size_t pNumberOfInstances) {
    for (size_t i = 0; i < mInverseIndex->size(); ++i) {
        mInverseIndex->operator[](i)->reserve(pNumberOfInstances);
    } 
}
void InverseIndexStorageUnorderedMap::insert(size_t pVectorId, size_t pHashValue, size_t pInstance, size_t pRemoveValueWithLeastSigificantBit) {

    if (pVectorId >= mInverseIndex->size()) {
        return;
    } 
    if (pRemoveValueWithLeastSigificantBit) {
        size_t leastSignificantBits = 0b11111111111111111111111111111111 << pRemoveValueWithLeastSigificantBit;
        size_t insertValue = pHashValue | leastSignificantBits;
        if (insertValue == leastSignificantBits) {
            return;
        }
    }
    if ((*mInverseIndex)[pVectorId] == NULL) return;
#ifdef OPENMP
#pragma omp critical
#endif
   {
        
        auto itHashValue_InstanceVector = (*mInverseIndex)[pVectorId]->find(pHashValue);

        // if for hash function h_i() the given hash values is already stored
        if (itHashValue_InstanceVector != (*mInverseIndex)[pVectorId]->end()) {
            // insert the instance id if not too many collisions (maxBinSize)
            if (itHashValue_InstanceVector->second->size() && itHashValue_InstanceVector->second->size() < mMaxBinSize) {
                // insert only if there wasn't any collisions in the past
                if (itHashValue_InstanceVector->second->size() > 0) {
                // std::cout << "Adding to vector" << std::endl;
                    
                    itHashValue_InstanceVector->second->push_back(pInstance);
                }
            } else { 
                // too many collisions: delete stored ids. empty vector is interpreted as an error code 
                // for too many collisions
                itHashValue_InstanceVector->second->clear();
            }
        } else {
            // given hash value for the specific hash function was not avaible: insert new hash value
            vsize_t* instanceIdVector = new vsize_t(1);
            (*instanceIdVector)[0] = pInstance;
            (*mInverseIndex)[pVectorId]->operator[](pHashValue) = instanceIdVector;
        }
   }
}

vsize_t* InverseIndexStorageUnorderedMap::getElement(size_t pVectorId, size_t pHashValue) {


    if (pVectorId < mInverseIndex->size() && (*mInverseIndex)[pVectorId] != NULL) {
                (*mInverseIndex)[pVectorId];
        auto iterator = (*mInverseIndex)[pVectorId]->find(pHashValue);
        if (iterator != (*mInverseIndex)[pVectorId]->end() && iterator->second != NULL) {
            return iterator->second;
        }
    }
    
	return NULL; 
}

distributionInverseIndex* InverseIndexStorageUnorderedMap::getDistribution() {
    distributionInverseIndex* retVal = new distributionInverseIndex();
    std::map<size_t, size_t> distribution;
    vsize_t numberOfCreatedHashValuesPerHashFunction;
    vsize_t averageNumberOfValuesPerHashValue;
    vsize_t standardDeviationPerNumberOfValuesPerHashValue;
    size_t meanForNumberHashValues = 0;
    
    for (auto it = mInverseIndex->begin(); it != mInverseIndex->end(); ++it) {
        if (*it == NULL) continue;
        numberOfCreatedHashValuesPerHashFunction.push_back((*it)->size());
        meanForNumberHashValues += (*it)->size();
        size_t mean = 0;
        
        for (auto itMap = (*it)->begin(); itMap != (*it)->end(); ++itMap) {
            distribution[itMap->second->size()] += 1;
            mean += itMap->second->size();
        }
        if ((*it)->size() != 0 || mean != 0) {
            mean = mean / (*it)->size();       
        }
        averageNumberOfValuesPerHashValue.push_back(mean);
        
        size_t variance = 0;
        for (auto itMap = (*it)->begin(); itMap != (*it)->end(); ++itMap) {
            variance += pow(static_cast<int>(itMap->second->size()) - mean, 2);
        }
        
        variance = variance / mInverseIndex->size();
        int standardDeviation = sqrt(variance);
        standardDeviationPerNumberOfValuesPerHashValue.push_back(standardDeviation);
    }
    
    size_t varianceForNumberOfHashValues = 0;
    for (auto it = mInverseIndex->begin(); it != mInverseIndex->end(); ++it) {
        if (*it == NULL) continue;
       varianceForNumberOfHashValues += pow((*it)->size() - meanForNumberHashValues, 2);
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
        for (auto itMap = (*it)->begin(); itMap != (*it)->end(); ++itMap) {
            if (itMap->second != NULL && itMap->second->size() <= pValue) {
                elementsToDelete.push_back(itMap->first);
                delete itMap->second;
                itMap->second = NULL;
            }
        }
        for (size_t i = 0; i < elementsToDelete.size(); ++i) {
            (*it)->erase(elementsToDelete[i]);
        }
        elementsToDelete.clear();
    }
}

// if pRemoveHashFunctionWithLessEntriesAs == 0 remove every hash function 
// which has less entries than mean+standard deviation
// else: remove every hash function which has less entries than pRemoveHashFunctionWithLessEntriesAs
void InverseIndexStorageUnorderedMap::removeHashFunctionWithLessEntriesAs(size_t pRemoveHashFunctionWithLessEntriesAs) {
    if (pRemoveHashFunctionWithLessEntriesAs == 0) {
        size_t mean = 0;
        size_t variance = 0;
        for (size_t i = 0; i < mInverseIndex->size(); ++i) {
            mean += (*mInverseIndex)[i]->size();
        }
        mean = mean / mInverseIndex->size();
        for (size_t i = 0; i < mInverseIndex->size(); ++i) {
            variance += pow(static_cast<int>((*mInverseIndex)[i]->size()) - mean, 2);
        }
        variance = variance / mInverseIndex->size();
        size_t standardDeviation = sqrt(variance);
        for (size_t i = 0; i < mInverseIndex->size(); ++i) {
            if ((*mInverseIndex)[i]->size() < mean + standardDeviation) {
                for (auto it = (*mInverseIndex)[i]->begin(); it != (*mInverseIndex)[i]->end(); ++it) {
                    delete it->second;
                    it->second = NULL;
                }
                delete (*mInverseIndex)[i];
                (*mInverseIndex)[i] = NULL;
            }
        }
    } else {
        for (size_t i = 0; i < mInverseIndex->size(); ++i) {
            if ((*mInverseIndex)[i] != NULL && (*mInverseIndex)[i]->size() < pRemoveHashFunctionWithLessEntriesAs) {
                for (auto it = (*mInverseIndex)[i]->begin(); it != (*mInverseIndex)[i]->end(); ++it) {
                    if (it->second != NULL) {
                        delete it->second;
                        it->second = NULL;
                    }
                }
                if ((*mInverseIndex)[i] != NULL) {
                    delete  (*mInverseIndex)[i];
                    (*mInverseIndex)[i] = NULL;
                }
            }
        }
    }
}