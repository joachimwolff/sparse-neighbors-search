#include "inverseIndexStorage.h"
#include "bloomierFilter/bloomierFilter.h"

#ifndef INVERSE_INDEX_STORAGE_BLOOMIER_FILTER_H
#define INVERSE_INDEX_STORAGE_BLOOMIER_FILTER_H
class InverseIndexStorageBloomierFilter : public InverseIndexStorage {
  private:
	std::vector<BloomierFilter> mInverseIndex;
  public:
    InverseIndexStorageBloomierFilter(size_t pSizeOfInverseIndex);
	~InverseIndexStorageBloomierFilter();
  	size_t size();
	vsize_t* getElement(size_t pVectorId, size_t pHashValue);
	void insert(size_t pVectorId, size_t pHashValue);
};
#endif // INVERSE_INDEX_STORAGE_BLOOMIER_FILTER_H