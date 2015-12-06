#include "inverseIndexStorageBloomierFilter.h"

InverseIndexStorageBloomierFilter::InverseIndexStorageBloomierFilter(size_t pSizeOfInverseIndex, size_t pMaxBinSize) {
	mInverseIndex = new std::vector<BloomierFilter* > (pSizeOfInverseIndex);
	mMaxBinSize = pMaxBinSize;
	// mKeys = new vvsize_t(pSizeOfInverseIndex, vsize_t());
	mValues = new vector__umapVector(pSizeOfInverseIndex);


	// mValues = new std::vector< vvsize_t>(pSizeOfInverseIndex, vvsize_t());
	mM = 100;
	mK = 1;
	mQ = 14;
}
InverseIndexStorageBloomierFilter::~InverseIndexStorageBloomierFilter() {
	
}
size_t InverseIndexStorageBloomierFilter::size() {
	return mInverseIndex->size();
}
vsize_t* InverseIndexStorageBloomierFilter::getElement(size_t pVectorId, size_t pHashValue) {
    return (*mInverseIndex)[pVectorId]->get(pHashValue); 
}
void InverseIndexStorageBloomierFilter::insert(size_t pVectorId, size_t pHashValue, size_t pInstance) {
	// (*mKeys)[pVectorId]->push_back(pHashValue);
   auto itHashValue_InstanceVector = (*mValues)[pVectorId].find(pHashValue);
	// if for hash function h_i() the given hash values is already stored
	if (itHashValue_InstanceVector != (*mValues)[pVectorId].end()) {
		// insert the instance id if not too many collisions (maxBinSize)
		if (itHashValue_InstanceVector->second.size() < mMaxBinSize) {
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
		(*mValues)[pVectorId][pHashValue] = instanceIdVector;
	}
}

void InverseIndexStorageBloomierFilter::create() {
	for (size_t i = 0; i < mValues->size(); ++i) {
		vsize_t* keys = new vsize_t((*mValues)[i].size());

		for (auto it = (*mValues)[i].begin(); it != (*mValues)[i].end(); ++it) {
	        (*keys)[i] = it->first;
    	}
		(*mInverseIndex)[i] = new BloomierFilter(mM, mK, mQ, keys, 100);
        (*mInverseIndex)[i]->create(&(*mValues)[i], 0 );
    }
}