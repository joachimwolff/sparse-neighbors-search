#include "inverseIndex.h"
#include "bloomierFilter/bloomierFilter.h" 

#ifndef INVERSE_INDEX_BLOOMIER_FILTER_H
#define INVERSE_INDEX_BLOOMIER_FILTER_H
class InverseIndexBloomierFilter : public InverseIndex {
	private: 
		std::vector<BloomierFilter>* mInverseIndexBloomierFilter;
	public:
		InverseIndexBloomierFilter(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions);
		vsize_t* computeSignature(const SparseMatrixFloat* pRawData, const size_t pInstance);
  		umap_uniqueElement* computeSignatureMap(const SparseMatrixFloat* pRawData);
  		void fit(const SparseMatrixFloat* pRawData);
  		neighborhood* kneighbors(const umap_uniqueElement* signaturesMap, const int pNneighborhood, const bool pDoubleElementsStorageCount);
};
#endif // INVERSE_INDEX_BLOOMIER_FILTER_H