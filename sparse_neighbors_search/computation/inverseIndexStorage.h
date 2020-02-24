/**
 Copyright 2016, 2017, 2018, 2019, 2020 Joachim Wolff
 PhD Thesis

 Copyright 2015, 2016 Joachim Wolff
 Master Thesis
 Tutor: Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwigs-University Freiburg im Breisgau
**/
#include "typeDefinitions.h"
#ifndef INVERSE_INDEX_STORAGE_H
#define INVERSE_INDEX_STORAGE_H
class InverseIndexStorage {
  public:
    // virtual InverseIndexStorage() = 0;
    virtual ~InverseIndexStorage() = 0;
	virtual size_t size() const = 0;
	virtual const vsize_t* getElement(size_t pVectorId, size_t pHashValue) = 0;
	virtual void insert(size_t pVectorId, size_t pHashValue, size_t pInstance, size_t pRemoveValueWithLeastSigificantBit) = 0;
    virtual distributionInverseIndex* getDistribution() = 0;
    virtual void prune(size_t pValue) = 0;
    virtual void removeHashFunctionWithLessEntriesAs(size_t pRemoveHashFunctionWithLessEntriesAs) = 0;
    virtual vector__umapVector_ptr* getIndex() = 0;
    virtual void reserveSpaceForMaps(size_t pNumberOfInstances) = 0;
};
inline InverseIndexStorage::~InverseIndexStorage() { }
#endif // INVERSE_INDEX_STORAGE_H