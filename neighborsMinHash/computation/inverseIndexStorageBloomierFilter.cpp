#include "inverseIndexStorageBloomierFilter.h"
#ifdef OPENMP
#include <omp.h>
#endif
InverseIndexStorageBloomierFilter::InverseIndexStorageBloomierFilter(const size_t pSizeOfInverseIndex, const size_t pMaxBinSize, const size_t pMaximalFeatures) {
	mInverseIndex = new std::vector<BloomierFilter* > (pSizeOfInverseIndex);
	mMaxBinSize = pMaxBinSize;
	mM = 3500;
	mK = 5;
	mQ = 7;
}
InverseIndexStorageBloomierFilter::~InverseIndexStorageBloomierFilter() {
    for (auto it = mInverseIndex->begin(); it != mInverseIndex->end(); ++it) {
        delete *it;
    }
	delete mInverseIndex;
}
distributionInverseIndex* InverseIndexStorageBloomierFilter::getDistribution(){
    return new distributionInverseIndex();
}
void InverseIndexStorageBloomierFilter::prune(int pValue) {
    
}
void InverseIndexStorageBloomierFilter::removeHashFunctionWithLessEntriesAs(int pRemoveHashFunctionWithLessEntriesAs) {
    
}
size_t InverseIndexStorageBloomierFilter::size() const {
	return mInverseIndex->size();
}
const vsize_t InverseIndexStorageBloomierFilter::getElement(size_t pVectorId, size_t pHashValue) {
    // return (*mInverseIndex)[pVectorId]->get(pHashValue); 
    vsize_t foo;
    return foo;
}
void InverseIndexStorageBloomierFilter::insert(size_t pVectorId, size_t pHashValue, size_t pInstance) {
    if ((*mInverseIndex)[pVectorId] == NULL) {
        (*mInverseIndex)[pVectorId] = new BloomierFilter(mM, mK, mQ, 100, mMaxBinSize);
    }
#ifdef OPENMP
#pragma omp critical
#endif
    (*mInverseIndex)[pVectorId]->set(pHashValue, pInstance);
}