#include "inverseIndexStorage.h"
#include "bloomierFilter/bloomierFilter.h"

#ifndef INVERSE_INDEX_STORAGE_BLOOMIER_FILTER_H
#define INVERSE_INDEX_STORAGE_BLOOMIER_FILTER_H
class InverseIndexStorageBloomierFilter : public InverseIndexStorage {
  private:
	std::vector<BloomierFilter* >* mInverseIndex;
	size_t mMaxBinSize;
	size_t mM;
	size_t mK;
	size_t mQ;
  public:
    InverseIndexStorageBloomierFilter(const size_t pSizeOfInverseIndex, const size_t pMaxBinSize, const size_t pMaximalFeatures);
	~InverseIndexStorageBloomierFilter();
  	size_t size() const;
	const vsize_t* getElement(size_t pVectorId, size_t pHashValue);
	void insert(size_t pVectorId, size_t pHashValue, size_t pInstance);
    std::map<size_t, size_t>* getDistribution();
    void prune(int pValue);
};
#endif // INVERSE_INDEX_STORAGE_BLOOMIER_FILTER_H