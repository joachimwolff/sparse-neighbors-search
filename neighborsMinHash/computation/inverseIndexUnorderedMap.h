#include "inverseIndex.h"

#ifndef INVERSE_INDEX_UNORDERED_MAP_H
#define INVERSE_INDEX_UNORDERED_MAP_H

class InverseIndexUnorderedMap : public InverseIndex {
	private: 
		std::vector<umapVector >* mInverseIndexUmapVector;
	public:
		InverseIndexUnorderedMap(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions);
	 	~InverseIndexUnorderedMap();
		vsize_t* computeSignature(const SparseMatrixFloat* pRawData, const size_t pInstance);
  		umap_uniqueElement* computeSignatureMap(const SparseMatrixFloat* pRawData);
  		void fit(const SparseMatrixFloat* pRawData);
  		neighborhood* kneighbors(const umap_uniqueElement* signaturesMap, const int pNneighborhood, const bool pDoubleElementsStorageCount);
};
#endif // INVERSE_INDEX_UNORDERED_MAP_H