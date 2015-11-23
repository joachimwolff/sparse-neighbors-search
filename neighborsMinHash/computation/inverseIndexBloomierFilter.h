#include "inverseIndex.h"
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
};
#endif // INVERSE_INDEX_BLOOMIER_FILTER_H