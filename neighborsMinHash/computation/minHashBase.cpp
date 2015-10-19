#include "minHashBase.h"

MinHashBase::MinHashBase(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions) {

      mInverseIndex = new InverseIndex(pNumberOfHashFunctions, pBlockSize,
                                      pNumberOfCores, pChunkSize,
                                      pMaxBinSize, pSizeOfNeighborhood, 
                                      pMinimalBlocksInCommon, pExcessFactor,
                                      pMaximalNumberOfHashCollisions);

}

MinHashBase::~MinHashBase(){
	delete mInverseIndex;
	delete mOriginalData;
}

void MinHashBase::fit(umapVector* instanceFeatureVector) {
	mInverseIndex->fit(instanceFeatureVector);
	return;
}

void MinHashBase::partialFit() {

}
neighborhood MinHashBase::kneighbors(rawData pRawData, size_t pNneighbors, size_t pReturnDistance=1, size_t pFast=0) {
	neighborhood neighborhood_ = mInverseIndex.kneighborhood(pRawData.inverseIndexData);
	if (fast) {
		return cutNeighborhood(&neighborhood_, pNneighbors);
	}



	real

	size_t appendSize = 0;
	for (size_t i = 0; i < neighborhood_.neighbor.size(); ++i) {
		size_t appendSize = nNeighbors - neighborhood_.neighbor[i].size();
		if (appendSize > 0) {
			
		}
	}
}
neighborhood MinHashBase::kneighborsGraph() {

}
neighborhood MinHashBase::fitKneighbors() {

}
neighborhood MinHashBase::fitKneighborsGraph() {

}

neighborhood MinHashBase::computeNeighborhood() {

}
neighborhood MinHashBase::computeExactNeighborhood() {

}
neighborhood MinHashBase::computeNeighborhoodGraph() {

}

void MinHashBase::cutNeighborhood(neighborhood* pNeighborhood, size_t pKneighborhood) {

}

