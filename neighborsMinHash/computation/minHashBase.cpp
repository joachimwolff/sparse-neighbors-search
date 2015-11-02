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
    neighborhood* neighborhood_ = mInverseIndex->kneighbors(X, pNneighbors, doubleElementsStorageCount);

    // if (pRawData->size() == 0) {

    //     for (auto it = X->begin(); it != X->end(); ++it) {
    //         delete it->second->instances;
    //         delete it->second->signature;
    //     }
    //     delete X;
    // }
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

#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
    for (size_t i = 0; i < neighborhood_->neighbors->size(); ++i) {

        SparseMatrixFloat* subMatrix = mOriginalData->getSubMatrixByRowVector(neighborhood_->neighbors->operator[](i));
        std::vector<sortMapFloat>* exactNeighbors = subMatrix->multiplyVectorAndSort(mOriginalData->getRow(i));
        std::vector<int> neighborsVector(exactNeighbors->size());
        std::vector<float> distancesVector(exactNeighbors->size());
        // std::cout << "exactNeighbors: ";
        for (size_t j = 0; j < exactNeighbors->size(); ++j) {
            // std::cout << (*exactNeighbors)[j].key << " ";
            neighborsVector[j] = static_cast<int> (neighborhood_->neighbors->operator[](i)[(*exactNeighbors)[j].key]);
            distancesVector[j] = (*exactNeighbors)[j].val;
        }
        // std::cout << "\nneighborhood_: ";
        // for (size_t j = 0; j < neighborhood_->neighbors->operator[](i).size(); ++j) {
        //     std::cout << neighborhood_->neighbors->operator[](i)[j] << " ";
        // }
        // std::cout << "\nEXact version: ";
        // for (size_t j = 0; j < neighborsVector.size(); ++j) {
        //     std::cout << neighborsVector[j] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "EXact version distance: ";
        // for (size_t j = 0; j < neighborsVector.size(); ++j) {
        //     std::cout << distancesVector[j] << " ";
        // }
        // std::cout << std::endl << std::endl;
        // if (i == 10) {
        //     return NULL;
        // }

#pragma omp critical
        {
            neighborhoodExact->neighbors->operator[](i) = neighborsVector;
            neighborhoodExact->distances->operator[](i) = distancesVector;
        }
    }
    delete neighborhood_->neighbors;
    delete neighborhood_->distances;
    delete neighborhood_;
    return neighborhoodExact;
}
