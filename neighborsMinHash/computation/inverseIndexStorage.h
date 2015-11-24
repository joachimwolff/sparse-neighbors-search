#include "typeDefinitions.h"
#ifndef INVERSE_INDEX_STORAGE_H
#define INVERSE_INDEX_STORAGE_H
class InverseIndexStorage {
  public:
    InverseIndexStorage();
	virtual ~InverseIndexStorage();
	virtual size_t size();
	virtual vsize_t* getElement(size_t pVectorId, size_t pHashValue);
	virtual void insert(size_t pVectorId, size_t pHashValue, size_t pInstance);
	// virtual void create();
};
#endif // INVERSE_INDEX_STORAGE_H