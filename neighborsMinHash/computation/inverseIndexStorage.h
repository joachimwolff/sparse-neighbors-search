#include "typeDefinitions.h"
#ifndef INVERSE_INDEX_STORAGE_H
#define INVERSE_INDEX_STORAGE_H
class InverseIndexStorage {
  public:
	virtual size_t size() = 0;
	virtual vsize_t* getElement(size_t pVectorId, size_t pHashValue) = 0;
	virtual void insert(size_t pVectorId, size_t pHashValue, size_t pInstance) = 0;
};
#endif // INVERSE_INDEX_STORAGE_H