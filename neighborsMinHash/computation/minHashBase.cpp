/**
 Copyright 2015 Joachim Wolff
 Master Thesis
 Tutors: Milad Miladi, Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwigs-University Freiburg im Breisgau
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
                    int pFast, int pSimilarity, size_t pBloomierFilter) {

        mInverseIndex = new InverseIndex(pNumberOfHashFunctions, pBlockSize,
                                    pNumberOfCores, pChunkSize,
                                    pMaxBinSize, pMinimalBlocksInCommon, 
                                    pExcessFactor, pMaximalNumberOfHashCollisions, pBloomierFilter);
    
        mNneighbors = pSizeOfNeighborhood;
        mFast = pFast;
        mNumberOfCores = pNumberOfCores;
        mChunkSize = pChunkSize;
        mSimilarity = pSimilarity;
}

MinHashBase::~MinHashBase() {
    delete mInverseIndex;
    delete mOriginalData;
}

void MinHashBase::fit(const SparseMatrixFloat* pRawData) {
    mInverseIndex->fit(pRawData);
    return;
}

void MinHashBase::partialFit() {

}
neighborhood* MinHashBase::kneighbors(const SparseMatrixFloat* pRawData, size_t pNneighbors, int pFast, int pSimilarity) {
    if (pFast == -1) {
        pFast = mFast;
    } 
    if (pNneighbors == 0) {
        pNneighbors = mNneighbors;
    }
    if (pSimilarity == -1) {
        pSimilarity = mSimilarity;
    }

    umap_uniqueElement* X;
    bool doubleElementsStorageCount = false;
    if (pRawData == NULL) {
        // no query data given, use stored signatures
        X = mInverseIndex->getSignatureStorage();
        doubleElementsStorageCount = true;
    } else {
        X = mInverseIndex->computeSignatureMap(pRawData);
    }
    neighborhood* neighborhood_ = mInverseIndex->kneighbors(X, pNneighbors, doubleElementsStorageCount);

    if (pFast) {     
        return neighborhood_;
    }

    neighborhood* neighborhoodExact = new neighborhood();
    neighborhoodExact->neighbors = new vvint(neighborhood_->neighbors->size());
    neighborhoodExact->distances = new vvfloat(neighborhood_->neighbors->size());

if (mChunkSize <= 0) {
        mChunkSize = ceil(neighborhood_->neighbors->size() / static_cast<float>(mNumberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
#ifdef OPENMP
#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
#endif
    for (size_t i = 0; i < neighborhood_->neighbors->size(); ++i) {

        if (neighborhood_->neighbors->operator[](i).size() != 1) {
            std::vector<sortMapFloat>* exactNeighbors;
            if (neighborhood_->neighbors->operator[](i).size() != 0) {
                if (pSimilarity) {
                    exactNeighbors = 
                        mOriginalData->cosineSimilarity(neighborhood_->neighbors->operator[](i), neighborhood_->neighbors->operator[](i)[0], pNneighbors, pRawData);
                } else {
                    exactNeighbors = 
                        mOriginalData->euclidianDistance(neighborhood_->neighbors->operator[](i), neighborhood_->neighbors->operator[](i)[0], pNneighbors, pRawData);
                }
            } 
            std::vector<int> neighborsVector(exactNeighbors->size());
            std::vector<float> distancesVector(exactNeighbors->size());

            for (size_t j = 0; j < exactNeighbors->size(); ++j) {
                neighborsVector[j] = (*exactNeighbors)[j].key;
                distancesVector[j] = (*exactNeighbors)[j].val;
            }
#ifdef OPENMP
#pragma omp critical
#endif
            {
                neighborhoodExact->neighbors->operator[](i) = neighborsVector;
                neighborhoodExact->distances->operator[](i) = distancesVector;
            }
        } else {
#ifdef OPENMP
#pragma omp critical
#endif
            {
                neighborhoodExact->neighbors->operator[](i) = neighborhood_->neighbors->operator[](i);
                neighborhoodExact->distances->operator[](i) = neighborhood_->distances->operator[](i);
            }
        }
    }
    delete neighborhood_->neighbors;
    delete neighborhood_->distances;
    delete neighborhood_;

    return neighborhoodExact;
}
