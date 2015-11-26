#include "inverseIndexStorageUnorderedMap.h"

 
InverseIndexStorageUnorderedMap::InverseIndexStorageUnorderedMap(size_t pSizeOfInverseIndex, size_t pMaxBinSize) {
	mSignatureStorage = new vector__umapVector(pSizeOfInverseIndex);
	mMaxBinSize = pMaxBinSize;
	mKeys = new vvsize_t(pSizeOfInverseIndex, vsize_t());
	mValues = new vvsize_t(pSizeOfInverseIndex, vsize_t());
}
InverseIndexStorageUnorderedMap::~InverseIndexStorageUnorderedMap() {
	
}
size_t InverseIndexStorageUnorderedMap::size() {
	return mSignatureStorage->size();
}
vsize_t* InverseIndexStorageUnorderedMap::getElement(size_t pVectorId, size_t pHashValue) {
	auto iterator = (*mSignatureStorage)[pVectorId].find(pHashValue);
	if (iterator != (*mSignatureStorage)[pVectorId].end()) {
		return &(iterator->second);
	}
	return new vsize_t(0);
	
	// auto itHashValue_InstanceVector = mInverseIndexUmapVector->operator[](j).find((*signature)[j]);
    //             // if for hash function h_i() the given hash values is already stored
    //             if (itHashValue_InstanceVector != mInverseIndexUmapVector->operator[](j).end()) {
    //                 // insert the instance id if not too many collisions (maxBinSize)
    //                 if (itHashValue_InstanceVector->second.size()
}
void InverseIndexStorageUnorderedMap::insert(size_t pVectorId, size_t pHashValue, size_t pInstance) {
	// std::cout << "insert" << std::endl;
	auto itHashValue_InstanceVector = (*mSignatureStorage)[pVectorId].find(pHashValue);
	// if for hash function h_i() the given hash values is already stored
	if (itHashValue_InstanceVector != (*mSignatureStorage)[pVectorId].end()) {
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
		(*mSignatureStorage)[pVectorId][pHashValue] = instanceIdVector;
	}
}

// void InverseIndexStorageUnorderedMap::create() {

// }