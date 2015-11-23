#ifndef INVERSE_INDEX_UNORDERED_MAP_H
#define INVERSE_INDEX_UNORDERED_MAP_H
#include "inverseIndex.h"

class InverseIndexUnorderedMap : public InverseIndex {
	private: 
		std::vector<umapVector >* mInverseIndexUmapVector;
	public:
		InverseIndexUnorderedMap(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions);
};
#endif // INVERSE_INDEX_UNORDERED_MAP_H