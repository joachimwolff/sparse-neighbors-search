#include "minHashBase.h"
#include <iostream>
#include "sparseMatrix.h"

#ifdef OPENMP
#include <omp.h>
#endif
// struct sort_map_float {
//     size_t key;
//     float val;
// };
MinHashBase::MinHashBase(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions,
                    int pFast) {

        mInverseIndex = new InverseIndex(pNumberOfHashFunctions, pBlockSize,
                                        pNumberOfCores, pChunkSize,
                                        pMaxBinSize, pMinimalBlocksInCommon, 
                                        pExcessFactor, pMaximalNumberOfHashCollisions);
        mNneighbors = pSizeOfNeighborhood;
        mFast = pFast;
        mNumberOfCores = pNumberOfCores;
        mChunkSize = pChunkSize;

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
neighborhood MinHashBase::kneighbors(const rawData pRawData, size_t pNneighbors, int pFast) {

    // std::cout << "start of kneighbors in minHashBase." << std::endl;
    if (pFast == -1) {
        pFast = mFast;
    } 
    if (pNneighbors == 0) {
        pNneighbors = mNneighbors;
    }
    umap_uniqueElement* X;
    bool doubleElementsStorageCount = false;
    if (pRawData.inverseIndexData->size() == 0) {
        // no query data given, use stored signatures
        X = mInverseIndex->getSignatureStorage();
        doubleElementsStorageCount = true;
    } else {
        X = mInverseIndex->computeSignatureMap(pRawData.inverseIndexData);
    }
    // std::cout << "Computing neighbors..." << std::endl;
    neighborhood neighborhood_ = mInverseIndex->kneighbors(X, pNneighbors, doubleElementsStorageCount);
    // std::cout << "Computing neighbors... Done!" << std::endl;


    if (pFast) {     
        return neighborhood_;
    }
        // std::cout << "64" << std::endl;

    neighborhood neighborhoodExact;
        // std::cout << "67" << std::endl;
    neighborhoodExact.neighbors = new vvint(neighborhood_.neighbors->size());
    // neighborhoodExact.neighbors->resize(neighborhood_.neighbors->size());
        // std::cout << "70" << std::endl;

    neighborhoodExact.distances = new vvfloat(neighborhood_.neighbors->size());
    // ->resize(neighborhood_.neighbors->size());
        // std::cout << "73" << std::endl;

if (mChunkSize <= 0) {
        mChunkSize = ceil(neighborhood_.neighbors->size() / static_cast<float>(mNumberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif

#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
    for (size_t i = 0; i < neighborhood_.neighbors->size(); ++i) {
        // std::cout << "66" << std::endl;
        // std::cout << "neighborhood count: " << neighborhood_.neighbors->operator[](i).size() << std::endl;
        std::vector<sortMapFloat> exactNeighbors = 
        mOriginalData->getSubMatrixByRowVector(neighborhood_.neighbors->operator[](i))->multiplyVectorAndSort(mOriginalData->getRow(i));
        // std::cout << "69" << std::endl;
        std::vector<int> neighborsVector(exactNeighbors.size());
        std::vector<float> distancesVector(exactNeighbors.size());
        // std::cout << "85" << std::endl;

        for (size_t j = 0; j < exactNeighbors.size(); ++j) {
        // std::cout << "88" << std::endl;

            neighborsVector[j] = static_cast<int> (neighborhood_.neighbors->operator[](i)[exactNeighbors[j].key]);
        // std::cout << "91" << std::endl;

            distancesVector[j] = exactNeighbors[j].val;
        }
        // std::cout << "95" << std::endl;
#pragma omp critical
        {
        neighborhoodExact.neighbors->operator[](i) = neighborsVector;
        // std::cout << "98" << std::endl;

        neighborhoodExact.distances->operator[](i) = distancesVector;
        // std::cout << "101" << std::endl;
        }
    }
   return neighborhoodExact;
}
