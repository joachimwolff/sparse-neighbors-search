#include "typeDefinitions.h"
#ifndef INVERSE_INDEX_STORAGE_H
#define INVERSE_INDEX_STORAGE_H
class InverseIndexStorage {
  public:
	virtual size_t size() const = 0;
	virtual const vsize_t* getElement(size_t pVectorId, size_t pHashValue) = 0;
	virtual void insert(size_t pVectorId, size_t pHashValue, size_t pInstance) = 0;
    virtual std::map<size_t, size_t>* getDistribution() = 0;
    virtual void prune(int pValue) = 0;
};
#endif // INVERSE_INDEX_STORAGE_H