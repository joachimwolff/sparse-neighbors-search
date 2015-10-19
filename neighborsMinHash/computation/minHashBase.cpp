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
neighborhood MinHashBase::kneighbors(rawData pRawData, size_t pNneighbors, size_t pReturnDistance, size_t pFast) {
	neighborhood neighborhood_ = mInverseIndex->kneighbors(&(*pRawData.inverseIndexData));
	if (pFast) {
		cutNeighborhood(&neighborhood_, pNneighbors, false, false);
		return neighborhood_;
	}



	// real

	// size_t appendSize = 0;
	// for (size_t i = 0; i < neighborhood_.neighbor.size(); ++i) {
	// 	size_t appendSize = nNeighbors - neighborhood_.neighbor[i].size();
	// 	if (appendSize > 0) {
			
	// 	}
	// }
	neighborhood a;
	return a;
}
neighborhood MinHashBase::kneighborsGraph() {
	neighborhood a;
	return a;
}
neighborhood MinHashBase::fitKneighbors() {
	neighborhood a;
	return a;
}
neighborhood MinHashBase::fitKneighborsGraph() {
	neighborhood a;
	return a;
}

neighborhood MinHashBase::computeNeighborhood() {
	neighborhood a;
	return a;
}
neighborhood MinHashBase::computeExactNeighborhood() {
	neighborhood a;
	return a;
}
neighborhood MinHashBase::computeNeighborhoodGraph() {
	neighborhood a;
	return a;
}

void MinHashBase::cutNeighborhood(neighborhood* pNeighborhood, size_t pKneighborhood, 
											bool pRadiusNeighbors, bool pWithFirstElement) {

	if (pRadiusNeighbors) {

	} else {
		size_t appendSize = 0;
		for (size_t i = 0; i < pNeighborhood->neighbors->size(); ++i) {
			appendSize = pKneighborhood - pNeighborhood->neighbors[i].size();
			for (size_t j = 0; j < appendSize; ++j) {
				(pNeighborhood->neighbors[i]).push_back(-1);
				(pNeighborhood->distances[i]).push_back(0.0);
			}
			if (pWithFirstElement) {
				pNeighborhood->neighbors[i].erase(pNeighborhood->neighbors[i].begin()+pKneighborhood, 
													pNeighborhood->neighbors[i].end());
			} else {
				pNeighborhood->neighbors[i].erase(pNeighborhood->neighbors[i].begin());
				pNeighborhood->neighbors[i].erase(pNeighborhood->neighbors[i].begin()+pKneighborhood, 
													pNeighborhood->neighbors[i].end());
			}
		}
	}

}

