#include "inverseIndexStorage.h"

#ifndef INVERSE_INDEX_STORAGE_UNORDERED_MAP_H
#define INVERSE_INDEX_STORAGE_UNORDERED_MAP_H
class InverseIndexStorageUnorderedMap : public InverseIndexStorage {
  private:
	vector__umapVector mSignatureStorage;
  public:
    InverseIndexStorageUnorderedMap(size_t pSizeOfInverseIndex);
	~InverseIndexStorageUnorderedMap();
  	size_t size();
	vsize_t getElement(size_t pVectorId, size_t pHashValue);
	void insert(size_t pVectorId, size_t pHashValue);
};
#endif // INVERSE_INDEX_STORAGE_UNORDERED_MAP_H