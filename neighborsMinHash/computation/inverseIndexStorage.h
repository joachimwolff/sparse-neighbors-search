#include "typeDefinitions.h"

class InverseIndexStorage {
  public:
    virtual InverseIndexStorage();
	virtual ~InverseIndexStorage();
	virtual size_t size();
	virtual vsize_t getElement(size_t pVectorId, size_t pHashValue);
	virtual void insert(size_t pVectorId, size_t pHashValue);
}