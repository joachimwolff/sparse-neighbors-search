#include "bloomierFilter.h"


#include "typeDefinitions.h"


#ifndef INVERSE_INDEX_H
#define INVERSE_INDEX_H
class InverseIndex {

  private: 
  	const double A = sqrt(2) - 1;
	  //const size_t MAX_VALUE = 2147483647;
    size_t mNumberOfHashFunctions;
    size_t mBlockSize;
    size_t mNumberOfCores;
    size_t mChunkSize;
    size_t mMaxBinSize;
    size_t mMinimalBlocksInCommon;
    size_t mExcessFactor;
    size_t mMaximalNumberOfHashCollisions;
    
    size_t mDoubleElementsStorageCount = 0;
    size_t mDoubleElementsQueryCount = 0;

  	umap_uniqueElement* mSignatureStorage;
  	std::vector<umapVector >* mInverseIndexUmapVector;
  	std::vector<BloomierFilter>* mInverseIndexBloomierFilter;

  	size_t _size_tHashSimple(size_t key, size_t aModulo) {
  	    key = ~key + (key << 15);
  	    key = key ^ (key >> 12);
  	    key = key + (key << 2);
  	    key = key ^ (key >> 4);
  	    key = key * 2057;
  	    key = key ^ (key >> 16);
  	    return key % aModulo;
  	};

  public:
  	InverseIndex(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions);
  	~InverseIndex();
  	vsize_t computeSignature(const vsize_t& featureVector);
  	umap_uniqueElement* computeSignatureMap(const umapVector* instanceFeatureVector);
  	void fit(const umapVector* instanceFeatureVector);
  	neighborhood kneighbors(const umap_uniqueElement* signaturesMap, const int pNneighborhood, const bool pDoubleElementsStorageCount);
  	umap_uniqueElement* getSignatureStorage(){return mSignatureStorage;};
};
#endif // INVERSE_INDEX_H