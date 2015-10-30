/**
 Copyright 2015 Joachim Wolff
 Master Thesis
 Tutors: Milad Miladi, Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/

#include <iostream>

#include "minHashBase.h"
#include "sparseMatrix.h"

#ifdef OPENMP
#include <omp.h>
#endif

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
    // delete mInverseIndex;
    // delete mOriginalData;
}

void MinHashBase::fit(const SparseMatrixFloat* pRawData) {
    mInverseIndex->fit(pRawData);
    return;
}

void MinHashBase::partialFit() {

}
neighborhood* MinHashBase::kneighbors(const SparseMatrixFloat* pRawData, size_t pNneighbors, int pFast) {
    if (pFast == -1) {
        pFast = mFast;
    } 
    if (pNneighbors == 0) {
        pNneighbors = mNneighbors;
    }
    umap_uniqueElement* X;
    bool doubleElementsStorageCount = false;
    if (pRawData->size() == 0) {
        // no query data given, use stored signatures
        X = mInverseIndex->getSignatureStorage();
        doubleElementsStorageCount = true;
    } else {
        X = mInverseIndex->computeSignatureMap(pRawData);
    }
    // std::cout <<  "69" << std::endl;

    neighborhood* neighborhood_ = mInverseIndex->kneighbors(X, pNneighbors, doubleElementsStorageCount);
    // std::cout <<  "72" << std::endl;

    if (pRawData->size() == 0) {
    // std::cout <<  "75" << std::endl;

        for (auto it = X->begin(); it != X->end(); ++it) {
            // delete it->second->instances;
            // delete it->second->signature;
        }
    // std::cout <<  "81" << std::endl;

        // delete X;
    }
    if (pFast) {     
    // std::cout <<  "86" << std::endl;

        return neighborhood_;
    }
    // std::cout <<  "80" << std::endl;
    neighborhood* neighborhoodExact = new neighborhood();
    neighborhoodExact->neighbors = new vvint(neighborhood_->neighbors->size());
    neighborhoodExact->distances = new vvfloat(neighborhood_->neighbors->size());
    // std::cout <<  "94" << std::endl;

if (mChunkSize <= 0) {
        mChunkSize = ceil(neighborhood_->neighbors->size() / static_cast<float>(mNumberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
    // std::cout <<  "92" << std::endl;

#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
    for (size_t i = 0; i < neighborhood_->neighbors->size(); ++i) {
    // std::cout <<  "96" << std::endl;

        SparseMatrixFloat* subMatrix = mOriginalData->getSubMatrixByRowVector(neighborhood_->neighbors->operator[](i));
    // std::cout <<  "99" << std::endl;

        std::vector<sortMapFloat>* exactNeighbors = subMatrix->multiplyVectorAndSort(mOriginalData->getRow(i));
    // std::cout <<  "102" << std::endl;

        // delete subMatrix;
        std::vector<int> neighborsVector(exactNeighbors->size());
        std::vector<float> distancesVector(exactNeighbors->size());
    // std::cout <<  "107" << std::endl;

        for (size_t j = 0; j < exactNeighbors->size(); ++j) {
            neighborsVector[j] = static_cast<int> (neighborhood_->neighbors->operator[](i)[(*exactNeighbors)[j].key]);
            distancesVector[j] = (*exactNeighbors)[j].val;
    // std::cout <<  "112" << std::endl;

        }
        // delete exactNeighbors;
    // std::cout <<  "116" << std::endl;

#pragma omp critical
        {
            neighborhoodExact->neighbors->operator[](i) = neighborsVector;
            neighborhoodExact->distances->operator[](i) = distancesVector;
        }
    // std::cout <<  "123" << std::endl;

    }
    // std::cout <<  "126" << std::endl;

    return neighborhoodExact;
}
