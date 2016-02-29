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
#include <set>
#include "minHashBase.h"
#include "sparseMatrix.h"

#ifdef OPENMP
#include <omp.h>
#endif

MinHashBase::MinHashBase(){};
MinHashBase::MinHashBase(size_t pNumberOfHashFunctions, size_t pShingleSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions,
                    int pFast, int pSimilarity, size_t pBloomierFilter,
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance, 
                    int pRemoveHashFunctionWithLessEntriesAs, size_t pHashAlgorithm,
                    size_t pBlockSize, size_t pShingle, size_t pRemoveValueWithLeastSigificantBit) {
            // printf("foo%i\n", __LINE__);

        mInverseIndex = new InverseIndex(pNumberOfHashFunctions, pShingleSize,
                                    pNumberOfCores, pChunkSize,
                                    pMaxBinSize, pMinimalBlocksInCommon, 
                                    pExcessFactor, pMaximalNumberOfHashCollisions, pBloomierFilter,
                                    pPruneInverseIndex, pPruneInverseIndexAfterInstance, 
                                    pRemoveHashFunctionWithLessEntriesAs, pHashAlgorithm, pBlockSize, pShingle,
                                    pRemoveValueWithLeastSigificantBit);

        mNneighbors = pSizeOfNeighborhood;
        mFast = pFast;
        mNumberOfCores = pNumberOfCores;
        mChunkSize = pChunkSize;
        mSimilarity = pSimilarity;
        mExcessFactor = pExcessFactor;
                    // printf("foo%i\n", __LINE__);

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
    // std::cout << "kneighbors" << std::endl;
    std::cout << "searching minhashBase" << std::endl;
    
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
        X = (mInverseIndex->computeSignatureMap(pRawData));
    }
    neighborhood* neighborhood_ = mInverseIndex->kneighbors(X, pNneighbors, 
                                                            doubleElementsStorageCount,
                                                            pFast, pSimilarity);
    std::cout << __LINE__ << std::endl;

                                                            
    if (!doubleElementsStorageCount) {
        delete X;
    }
    if (pFast) {     
        return neighborhood_;
    }
    #ifdef CUDA
        return neighborhood_;
    #endif

    neighborhood* neighborhoodExact = new neighborhood();
    neighborhoodExact->neighbors = new vvint(neighborhood_->neighbors->size());
    neighborhoodExact->distances = new vvfloat(neighborhood_->neighbors->size());
    std::cout << __LINE__ << std::endl;


if (mChunkSize <= 0) {
        mChunkSize = ceil(neighborhood_->neighbors->size() / static_cast<float>(mNumberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
    vvint neighborsListFirstRound(neighborhood_->neighbors->size());
    for (size_t i = 0; i < neighborsListFirstRound.size(); ++i) {
        neighborsListFirstRound[i] = vint(0);
    }
    size_t numberOfRounds = 2;
    for (size_t round = 0; round < numberOfRounds; ++round) {
    #ifdef OPENMP
    #pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
    #endif
        for (size_t i = 0; i < neighborhood_->neighbors->size(); ++i) {
            if (neighborhood_->neighbors->operator[](i).size() > 0) {
                std::vector<sortMapFloat> exactNeighbors;
                // if (0 < neighborhood_->neighbors->operator[](i).size()) {
                    if (pSimilarity) {
                        exactNeighbors = 
                            mOriginalData->cosineSimilarity(neighborhood_->neighbors->operator[](i), pNneighbors+mExcessFactor, pRawData);
                    } else {
                        exactNeighbors = 
                            mOriginalData->euclidianDistance(neighborhood_->neighbors->operator[](i), pNneighbors+mExcessFactor, pRawData);
                    }
                    
                    size_t vectorSize = exactNeighbors.size();
                    
                    for (size_t j = 0; j < vectorSize && j < pNneighbors+mExcessFactor; ++j) {
    // #ifdef OPENMP
    // #pragma omp critical
    // #endif
                        neighborsListFirstRound[i].push_back(exactNeighbors[j].key);
                    }
                // }
            } else {
                neighborsListFirstRound[i].clear();
            }
        }
        
    #ifdef OPENMP
    #pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
    #endif   
        for (size_t i = 0; i < neighborsListFirstRound.size(); ++i) {
            size_t sizeOfExtended = neighborsListFirstRound[i].size();
            vsize_t dublicateElements((neighborhood_->neighbors->size()/sizeof(size_t))+1,0);
            size_t bucketIndex;
            size_t element;
            size_t valueSeen;
            size_t instance;
            
            neighborhood_->neighbors->operator[](i).clear();
            for (size_t j = 0; j < pNneighbors && j < sizeOfExtended; ++j) { 
                instance = neighborsListFirstRound[i][j];
                // if (instance > neighborsListFirstRound.size()) continue;
                // if (neighborsListFirstRound[instance].size() > 1000) continue;
                for (size_t k = 0; k < neighborsListFirstRound[instance].size();  ++k) {
                    bucketIndex = neighborsListFirstRound[instance][k] / sizeof(size_t);
                    element = 1 << (neighborsListFirstRound[instance][k] % sizeof(size_t));
                    // if (bucketIndex > dublicateElements.size()) continue;
                    valueSeen = dublicateElements[bucketIndex] & element;
                    
                    if (valueSeen != element) {
                    // #ifdef OPENMP
                    // #pragma omp critical
                    // #endif
                        neighborhood_->neighbors->operator[](i).push_back(neighborsListFirstRound[instance][k]);
                        dublicateElements[bucketIndex] = dublicateElements[bucketIndex] | element;
                    }
                }
            }
        }
    }
    
 #ifdef OPENMP
#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
#endif   
    for (size_t i = 0; i < neighborhood_->neighbors->size(); ++i) {
       if (neighborhood_->neighbors->operator[](i).size() != 1) {
                std::vector<sortMapFloat> exactNeighbors;
                if (0 < neighborhood_->neighbors->operator[](i).size()) {
                    if (pSimilarity) {
                        exactNeighbors = 
                            mOriginalData->cosineSimilarity(neighborhood_->neighbors->operator[](i), pNneighbors, pRawData);
                    } else {
                        exactNeighbors = 
                            mOriginalData->euclidianDistance(neighborhood_->neighbors->operator[](i), pNneighbors, pRawData);
                    }
                }
            size_t vectorSize = exactNeighbors.size();
            
            std::vector<int> neighborsVector(vectorSize);
            std::vector<float> distancesVector(vectorSize);
            if (vectorSize == 0) {
                neighborsVector.push_back(i);
                distancesVector.push_back(0.0);
            }
            for (size_t j = 0; j < vectorSize; ++j) {
                    neighborsVector[j] = exactNeighbors[j].key;
                    distancesVector[j] = exactNeighbors[j].val;
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
                std::vector<int> neighborsVector(1, i);
                std::vector<float> distancesVector(1, 0.0);
                neighborhoodExact->neighbors->operator[](i) = neighborsVector;
                neighborhoodExact->distances->operator[](i) = distancesVector;
            }
        }
    }
    
    delete neighborhood_->neighbors;
    delete neighborhood_->distances;
    delete neighborhood_;

    return neighborhoodExact;
}

distributionInverseIndex* MinHashBase::getDistributionOfInverseIndex() {
    return mInverseIndex->getDistribution();
}
