#include "inverseIndexStorageBloomierFilter.h"

InverseIndexStorageBloomierFilter::InverseIndexStorageBloomierFilter(size_t pSizeOfInverseIndex, size_t pMaxBinSize) {
	mInverseIndex = new std::vector<BloomierFilter* > (pSizeOfInverseIndex);
	mMaxBinSize = pMaxBinSize;
	mKeys = new vvsize_t(pSizeOfInverseIndex, vsize_t());
	mValues = new vvsize_t(pSizeOfInverseIndex, vsize_t());
	mM = 100;
	mK = 100;
	mQ = 100;
	
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
		(*mInverseIndex)[pVectorId] = new BloomierFilter(mM, mK, mQ);
	}
	// std::cout << "26" << std::endl;
	(*mInverseIndex)[pVectorId]->set(pHashValue, pInstance);
	// std::cout << "28" << std::endl;
	
	// (*mKeys)[pVectorId].push_back(pHashValue);
	// (*mValues)[pVectorId].push_back(pInstance);
}
// void InverseIndexStorageBloomierFilter::create() {
		
// 	for (size_t i = 0; i < mKeys->size(); ++i) {
// 		BloomierFilter* bloomierFilter = new BloomierFilter(m, k, q);
// 		bloomierFilter->create((*mKeys)[i], &(*mValues)[i]);
// 		(*mInverseIndex)[i] = bloomierFilter;
// 	}
// }