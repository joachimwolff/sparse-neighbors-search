#include "inverseIndexStorage.h"

#ifndef INVERSE_INDEX_STORAGE_UNORDERED_MAP_H
#define INVERSE_INDEX_STORAGE_UNORDERED_MAP_H
class InverseIndexStorageUnorderedMap : public InverseIndexStorage {
  private:
	vector__umapVector* mSignatureStorage;
	size_t mMaxBinSize;
	vvsize_t* mKeys;
	vvsize_t* mValues;
  public:
    InverseIndexStorageUnorderedMap(size_t pSizeOfInverseIndex, size_t pMaxBinSize);
	~InverseIndexStorageUnorderedMap();
  	size_t size();
	vsize_t* getElement(size_t pVectorId, size_t pHashValue);
	void insert(size_t pVectorId, size_t pHashValue, size_t pInstance);
	// void create();
};
#endif // INVERSE_INDEX_STORAGE_UNORDERED_MAP_H