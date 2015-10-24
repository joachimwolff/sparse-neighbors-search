#include "minHashBase.h"
#include <iostream>
MinHashBase::MinHashBase(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions) {

      mInverseIndex = new InverseIndex(pNumberOfHashFunctions, pBlockSize,
                                      pNumberOfCores, pChunkSize,
                                      pMaxBinSize, pMinimalBlocksInCommon, 
                                      pExcessFactor, pMaximalNumberOfHashCollisions);
      mNneighbors = pSizeOfNeighborhood;

}

MinHashBase::~MinHashBase(){
	delete mInverseIndex;
	// delete mOriginalData;
}

void MinHashBase::fit(umapVector* instanceFeatureVector) {
	mInverseIndex->fit(instanceFeatureVector);
	return;
}

void MinHashBase::partialFit() {

}
neighborhood MinHashBase::kneighbors(rawData pRawData, size_t pNneighbors, size_t pFast) {
	std::cout << "start of kneighbors in minHashBase." << std::endl;
	if (pNneighbors == 0) {
		pNneighbors = mNneighbors;
	}
	umap_uniqueElement* X;
	if (pRawData.inverseIndexData->size() == 0) {
		// no query data given, use stored signatures
		X = mInverseIndex->getSignatureStorage();
	} else {
		X = mInverseIndex->computeSignatureMap(pRawData.inverseIndexData);
	}
	std::cout << "Computing neighbors..." << std::endl;
	neighborhood neighborhood_ = mInverseIndex->kneighbors(X, pNneighbors);
	std::cout << "Computing neighbors... Done!" << std::endl;


	if (pFast) {		
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

