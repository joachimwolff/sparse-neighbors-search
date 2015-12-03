#include "inverseIndexStorageBloomierFilter.h"

InverseIndexStorageBloomierFilter::InverseIndexStorageBloomierFilter(size_t pSizeOfInverseIndex, size_t pMaxBinSize) {
	mInverseIndex = new std::vector<BloomierFilter* > (pSizeOfInverseIndex);
	// mMaxBinSize = pMaxBinSize;
	// mKeys = new vvsize_t(pSizeOfInverseIndex, vsize_t());
    mValues = new std::vector< std::unordered_map<size_t, vsize_t> >(pSizeOfInverseIndex, std::unordered_map<size_t, vsize_t()>);

	// mValues = new std::vector< vvsize_t>(pSizeOfInverseIndex, vvsize_t());
	mM = 50;
	mK = 5;
	mQ = 13;
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
    if ( (*mValues)[pVectorId].find(pHashValue) !=  (*mValues)[pVectorId].end()) {
        
    }
    // (*mValues)[pVectorId][pHashValue]
	// (*mValues)[pVectorId]->push_back(pInstance);
	// if ((*mInverseIndex)[pVectorId] == NULL) {
		// (*mInverseIndex)[pVectorId] = new BloomierFilter(mM, mK, mQ, 100);
	// }
	// (*mInverseIndex)[pVectorId]->set(pHashValue, pInstance);
}

void InverseIndexStorageBloomierFilter::create() {
	for (size_t i = 0; i < mKeys->size(); ++i) {
		(*mInverseIndex)[pVectorId] = new BloomierFilter(mM, mK, mQ, 100);
        (*mInverseIndex)[pVectorId]->create((*mValues)[i], 0 );
    }
}