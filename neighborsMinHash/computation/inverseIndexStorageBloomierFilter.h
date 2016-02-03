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
	void insert(size_t pVectorId, size_t pHashValue, size_t pInstance, size_t pRemoveValueWithLeastSigificantBit);
    distributionInverseIndex* getDistribution();
    void prune(size_t pValue);
    void removeHashFunctionWithLessEntriesAs(size_t pRemoveHashFunctionWithLessEntriesAs);
};
#endif // INVERSE_INDEX_STORAGE_BLOOMIER_FILTER_H