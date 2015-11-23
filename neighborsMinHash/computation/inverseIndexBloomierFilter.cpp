#include "inverseIndexBloomierFilter.h"

InverseIndexBloomierFilter::InverseIndexBloomierFilter(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions) : InverseIndex(pNumberOfHashFunctions, pBlockSize,
                    pNumberOfCores, pChunkSize,
                    pMaxBinSize, pMinimalBlocksInCommon,
                    pExcessFactor, pMaximalNumberOfHashCollisions) {
		mInverseIndexBloomierFilter = new std::vector<BloomierFilter>();
  
}

vsize_t* InverseIndexBloomierFilter::computeSignature(const SparseMatrixFloat* pRawData, const size_t pInstance) {
    
}
umap_uniqueElement* InverseIndexBloomierFilter::computeSignatureMap(const SparseMatrixFloat* pRawData) {
    
}
void InverseIndexBloomierFilter::fit(const SparseMatrixFloat* pRawData) {
    
}
neighborhood* InverseIndexBloomierFilter::kneighbors(const umap_uniqueElement* signaturesMap, 
                                                    const int pNneighborhood, const bool pDoubleElementsStorageCount) {
                                                    
    
}