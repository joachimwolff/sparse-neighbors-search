#include "inverseIndexStorageBloomierFilter.h"

InverseIndexStorageBloomierFilter::InverseIndexStorageBloomierFilter(size_t pSizeOfInverseIndex, size_t pMaxBinSize) {
	mInverseIndex = new std::vector<BloomierFilter* > (pSizeOfInverseIndex);
	mMaxBinSize = pMaxBinSize;
	mM = 10000;
	mK = 5;
	mQ = 8;
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
    if ((*mInverseIndex)[pVectorId] == NULL) {
        (*mInverseIndex)[pVectorId] = new BloomierFilter(mM, mK, mQ, 100, mMaxBinSize);
    }
    (*mInverseIndex)[pVectorId]->set(pHashValue, pInstance);
}