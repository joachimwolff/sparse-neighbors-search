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
#include "inverseIndexStorage.h"

#ifndef INVERSE_INDEX_STORAGE_UNORDERED_MAP_H
#define INVERSE_INDEX_STORAGE_UNORDERED_MAP_H
class InverseIndexStorageUnorderedMap {
  private:
	vector__umapVector_ptr* mInverseIndex = NULL;
	size_t mMaxBinSize;
	// vvsize_t* mKeys;
	// vvsize_t* mValues;
  public:
    InverseIndexStorageUnorderedMap(size_t pSizeOfInverseIndex, size_t pMaxBinSize);
	~InverseIndexStorageUnorderedMap();
  	size_t size() const;
	vsize_t* getElement(size_t pVectorId, size_t pHashValue);
	void insert(size_t pVectorId, size_t pHashValue, size_t pInstance, size_t pRemoveValueWithLeastSigificantBit);
    // void insert(vector__umapVector_ptr::iterator start, vector__umapVector_ptr::iterator end);
    distributionInverseIndex* getDistribution();
    void prune(size_t pValue);
    void removeHashFunctionWithLessEntriesAs(size_t pRemoveHashFunctionWithLessEntriesAs);
    vector__umapVector_ptr* getIndex() { return mInverseIndex;};
    void reserveSpaceForMaps(size_t pNumberOfInstances);
	// void create();
};
#endif // INVERSE_INDEX_STORAGE_UNORDERED_MAP_H