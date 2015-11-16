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
    delete mInverseIndex;
    delete mOriginalData;
}

void MinHashBase::fit(const SparseMatrixFloat* pRawData) {
    mInverseIndex->fit(pRawData);
    return;
}

void MinHashBase::partialFit() {

}
neighborhood* MinHashBase::kneighbors(const SparseMatrixFloat* pRawData, size_t pNneighbors, int pFast, size_t pSimilarity) {
    // std::cout << "53M" << std::endl;
    if (pFast == -1) {
        pFast = mFast;
    } 
    if (pNneighbors == 0) {
        pNneighbors = mNneighbors;
    }
    // std::cout << "61M" << std::endl;

    umap_uniqueElement* X;
    bool doubleElementsStorageCount = false;
    if (pRawData->size() == 0) {
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
    // std::cout << "77M" << std::endl;

    neighborhood* neighborhoodExact = new neighborhood();
    neighborhoodExact->neighbors = new vvint(neighborhood_->neighbors->size());
    neighborhoodExact->distances = new vvfloat(neighborhood_->neighbors->size());

if (mChunkSize <= 0) {
        mChunkSize = ceil(neighborhood_->neighbors->size() / static_cast<float>(mNumberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
    // std::cout << "89M" << std::endl;

#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
    for (size_t i = 0; i < neighborhood_->neighbors->size(); ++i) {
    // std::cout << "93M" << std::endl;

        if (neighborhood_->neighbors->operator[](i).size() != 1) {
    // std::cout << "96M" << std::endl;
    // std::cout << "i: " << i << std::endl;
    // std::cout << "size: " << neighborhood_->neighbors->size() << std::endl;
    // std::cout << "start foo: " << neighborhood_->neighbors->operator[](i).size() << std::endl ;
    // std::cout << "again foo: "<< neighborhood_->neighbors->operator[](i)[0] << std::endl;
            std::vector<sortMapFloat>* exactNeighbors;
            if (pSimilarity) {
                exactNeighbors = 
                    mOriginalData->cosineSimilarity(neighborhood_->neighbors->operator[](i), neighborhood_->neighbors->operator[](i)[0], pNneighbors);
            } else {
                exactNeighbors = 
                    mOriginalData->euclidianDistance(neighborhood_->neighbors->operator[](i), neighborhood_->neighbors->operator[](i)[0], pNneighbors);
            }
            
    // std::cout << "101M" << std::endl;
            std::vector<int> neighborsVector(exactNeighbors->size());
            std::vector<float> distancesVector(exactNeighbors->size());
    // std::cout << "104M" << std::endl;

            for (size_t j = 0; j < exactNeighbors->size(); ++j) {
                neighborsVector[j] = (*exactNeighbors)[j].key;
                distancesVector[j] = (*exactNeighbors)[j].val;
            }
    // std::cout << "105M" << std::endl;

#pragma omp critical
            {
    // std::cout << "109M" << std::endl;

                neighborhoodExact->neighbors->operator[](i) = neighborsVector;
                neighborhoodExact->distances->operator[](i) = distancesVector;
            }
        } else {
#pragma omp critical
            {
    // std::cout << "117M" << std::endl;

                neighborhoodExact->neighbors->operator[](i) = neighborhood_->neighbors->operator[](i);
                neighborhoodExact->distances->operator[](i) = neighborhood_->distances->operator[](i);
            }
        }
    }
    // std::cout << "124M" << std::endl;

    delete neighborhood_->neighbors;
    delete neighborhood_->distances;
    delete neighborhood_;
    // std::cout << "129M" << std::endl;

    return neighborhoodExact;
}
