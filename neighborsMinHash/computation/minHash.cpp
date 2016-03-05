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
#include "minHash.h"
#include "sparseMatrix.h"

#ifdef OPENMP
#include <omp.h>
#endif

MinHash::MinHash(){};
MinHash::MinHash(size_t pNumberOfHashFunctions, size_t pShingleSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions,
                    int pFast, int pSimilarity, int pPruneInverseIndex, float pPruneInverseIndexAfterInstance, 
                    int pRemoveHashFunctionWithLessEntriesAs, size_t pHashAlgorithm,
                    size_t pBlockSize, size_t pShingle, size_t pRemoveValueWithLeastSigificantBit,
                    float pCpuGpuLoadBalancing) {
            // printf("foo%i\n", __LINE__);

        mInverseIndex = new InverseIndex(pNumberOfHashFunctions, pShingleSize,
                                    pNumberOfCores, pChunkSize,
                                    pMaxBinSize, pMinimalBlocksInCommon, 
                                    pExcessFactor, pMaximalNumberOfHashCollisions,
                                    pPruneInverseIndex, pPruneInverseIndexAfterInstance, 
                                    pRemoveHashFunctionWithLessEntriesAs, pHashAlgorithm, pBlockSize, pShingle,
                                    pRemoveValueWithLeastSigificantBit, pCpuGpuLoadBalancing);

        mNneighbors = pSizeOfNeighborhood;
        mFast = pFast;
        mNumberOfCores = pNumberOfCores;
        mChunkSize = pChunkSize;
        mSimilarity = pSimilarity;
        mExcessFactor = pExcessFactor;
        mCpuGpuLoadBalancing = pCpuGpuLoadBalancing;
        mHash = new Hash();
                    // printf("foo%i\n", __LINE__);

}

MinHash::~MinHash() {
    delete mInverseIndex;
    delete mOriginalData;
    delete mHash;
}

void MinHash::fit(const SparseMatrixFloat* pRawData) {
    mInverseIndex->fit(pRawData);
    return;
}

void MinHash::partialFit() {

}
neighborhood* MinHash::kneighbors(const SparseMatrixFloat* pRawData, size_t pNneighbors, int pFast, int pSimilarity) {
    if (pFast == -1) {
        pFast = mFast;
    } 
    if (pNneighbors == 0) {
        pNneighbors = mNneighbors;
    }
    
    if (pSimilarity == -1) {
        pSimilarity = mSimilarity;
    }
    // handle cpu gpu load
    // mCpuGpuLoadBalancing
    bool doubleElementsStorageCount = false;
    
    neighborhood* neighborhood_;
    if (pRawData == NULL) {
        // no query data given, use stored signatures
        // neighborhood_ = neighborhood_inverseIndex;
        umap_uniqueElement* x_inverseIndex = mInverseIndex->getSignatureStorage();
        neighborhood_  = mInverseIndex->kneighbors(x_inverseIndex, 
                                                    pNneighbors, true,
                                                    pFast, pSimilarity);
        doubleElementsStorageCount = true;
    } else {
        umap_uniqueElement* X = (mInverseIndex->computeSignatureMap(pRawData));
        neighborhood_ = mInverseIndex->kneighbors(X, pNneighbors, 
                                                doubleElementsStorageCount,
                                                pFast, pSimilarity);
    
        for (auto it = X->begin(); it != X->end(); ++it) {
        // if ((*it).second.instances != NULL) {
            delete (*it).second.instances;
        // }
        // if ((*it).second.signature != NULL) {
            delete (*it).second.signature;
        // }
        }
        delete X;
    }

    if (pFast) {     
        return neighborhood_;
    }
    #ifdef CUDA
        return neighborhood_;
    #endif

if (mChunkSize <= 0) {
        mChunkSize = ceil(neighborhood_->neighbors->size() / static_cast<float>(mNumberOfCores));
    }
#ifdef OPENMP
    omp_set_dynamic(0);
#endif
    vvint neighborsListFirstRound(neighborhood_->neighbors->size(), vint(0));
   
    neighborhood* neighborhoodCandidates = new neighborhood();
    neighborhoodCandidates->neighbors = new vvint(mOriginalData->size());
    
    // compute the exact neighbors based on the candidates given by the inverse index
    // store the neighborhood in neighborhoodCandidates to reuse neighbor if needed
    // and store neighbors list per requested instance in neighborsListFirstRound
    //
    // neighborsListFirstRound is needed to get replace the candidates in neighborhood_
    // if it would not be used, it would be iterated over neighborhood_ and manipulated
    // at the same time
    #ifdef OPENMP
    #pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
    #endif
    for (size_t i = 0; i < neighborhood_->neighbors->size(); ++i) {
        if (neighborhood_->neighbors->operator[](i).size() > 0) {
            std::vector<sortMapFloat> exactNeighbors;
            if (pSimilarity) {
                exactNeighbors = 
                    mOriginalData->cosineSimilarity(neighborhood_->neighbors->operator[](i), pNneighbors+mExcessFactor, i, pRawData);
            } else {
                exactNeighbors = 
                    mOriginalData->euclidianDistance(neighborhood_->neighbors->operator[](i), pNneighbors+mExcessFactor, i, pRawData);
            }
            
            size_t vectorSize = std::min(exactNeighbors.size(),pNneighbors+mExcessFactor);
            
            std::vector<int> neighborsVector(vectorSize);
            for (size_t j = 0; j < vectorSize; ++j) {
                // std::cout << "key: " << exactNeighbors[j].key << ": " << exactNeighbors[j].val << std::endl;
                neighborsVector[j] = exactNeighbors[j].key;
                neighborsListFirstRound[i].push_back(exactNeighbors[j].key);
            } 
            #ifdef OPENMP
            #pragma omp critical
            #endif
            {
                if (exactNeighbors.size() > 1) {
                    neighborhoodCandidates->neighbors->operator[](exactNeighbors[0].key) = neighborsVector;
                }
            }
        }
    }
    umap_uniqueElement* x_inverseIndex = mInverseIndex->getSignatureStorage();
    #ifdef OPENMP
    #pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
    #endif   
    // for all requested instances get the neighbors+mExcessFactor of the neighbors
    for (size_t i = 0; i < neighborsListFirstRound.size(); ++i) {
        size_t sizeOfExtended = neighborsListFirstRound[i].size();
        vsize_t dublicateElements((mOriginalData->size()/sizeof(size_t))+1,0);
        size_t bucketIndex;
        size_t element;
        size_t valueSeen;
        size_t instance;
        size_t queryInstance;
        if (neighborhood_->neighbors->operator[](i).size() > 0) {
            queryInstance = neighborhood_->neighbors->operator[](i)[0];
        }
        neighborhood_->neighbors->operator[](i).clear();
        neighborhood_->neighbors->operator[](i).push_back(queryInstance); 
        // std::cout << "size: " << neighborhood_->neighbors->operator[](i).size() << " element: " << queryInstance << std::endl;
        bucketIndex = queryInstance / sizeof(size_t);
        element = 1 << queryInstance % sizeof(size_t);
        dublicateElements[bucketIndex] = dublicateElements[bucketIndex] | element;
        
        for (size_t j = 0; j < pNneighbors && j < sizeOfExtended; ++j) { 
            instance = neighborsListFirstRound[i][j];
            // neighborhood for instance was already computed?
            if (neighborhoodCandidates->neighbors->operator[](instance).size() == 0) {
                
                
                umap_uniqueElement* instance_signature = new umap_uniqueElement();
                
                size_t signatureId = 0;
                for (size_t k = 0; k < mOriginalData->getSizeOfInstance(instance); ++k) {
                        signatureId = mHash->hash((mOriginalData->getNextElement(instance, k) +1), (signatureId+1), MAX_VALUE);
                }
                
                (*instance_signature)[signatureId] = (*x_inverseIndex)[signatureId];
                 
                 neighborhood* neighborhood_instance = mInverseIndex->kneighbors(instance_signature, 
                                                                pNneighbors, false,
                                                                pFast, pSimilarity, false);
                                                    
                if (neighborhood_instance->neighbors->operator[](0).size() != 0) { 
                    std::vector<sortMapFloat> exactNeighbors;
                    if (pSimilarity) {
                        exactNeighbors = 
                            mOriginalData->cosineSimilarity(neighborhood_instance->neighbors->operator[](0), pNneighbors+mExcessFactor, instance, pRawData);
                    } else {
                        exactNeighbors = 
                            mOriginalData->euclidianDistance(neighborhood_instance->neighbors->operator[](0), pNneighbors+mExcessFactor, instance, pRawData);
                    }
                    size_t vectorSize = std::min(exactNeighbors.size(), pNneighbors+mExcessFactor);
                    std::vector<int> neighborsVector(vectorSize);
                    for (size_t j = 0; j < vectorSize; ++j) {
                        neighborsVector[j] = exactNeighbors[j].key;
                    }
                    #ifdef OPENMP
                    #pragma omp critical
                    #endif
                    {
                        
                        if (exactNeighbors.size() > 1) {
                            neighborhoodCandidates->neighbors->operator[](instance) = neighborsVector;
                        }
                    } 
                }
                
                // delete x_inverseIndex;
                delete instance_signature;
                delete neighborhood_instance;
            }
             
            // add the neighbors + mExcessFactor to the candidate list 
            for (size_t k = 0; k < neighborhoodCandidates->neighbors->operator[](instance).size() && k < pNneighbors+mExcessFactor; ++k) {
                bucketIndex = neighborhoodCandidates->neighbors->operator[](instance)[k] / sizeof(size_t);
                element = 1 << neighborhoodCandidates->neighbors->operator[](instance)[k] % sizeof(size_t);
                valueSeen = dublicateElements[bucketIndex] & element;
                // if candidate was already inserted, do not inserte it a second time
                if (valueSeen != element) {
                    neighborhood_->neighbors->operator[](i).push_back(neighborhoodCandidates->neighbors->operator[](instance)[k]);
                    dublicateElements[bucketIndex] = dublicateElements[bucketIndex] | element;
                }
            }
        }
    }

    delete neighborhoodCandidates->neighbors;
    delete neighborhoodCandidates;
    
    neighborhood* neighborhoodExact = new neighborhood();
    neighborhoodExact->neighbors = new vvint(neighborhood_->neighbors->size());
    neighborhoodExact->distances = new vvfloat(neighborhood_->neighbors->size());

    // compute the exact neighbors based on the candidate selection before.
#ifdef OPENMP
#pragma omp parallel for schedule(static, mChunkSize) num_threads(mNumberOfCores)
#endif   
    for (size_t i = 0; i < neighborhood_->neighbors->size(); ++i) {
       if (neighborhood_->neighbors->operator[](i).size() != 1) {
                std::vector<sortMapFloat> exactNeighbors;
                if (0 < neighborhood_->neighbors->operator[](i).size()) {
                    if (pSimilarity) {
                        exactNeighbors = 
                            mOriginalData->cosineSimilarity(neighborhood_->neighbors->operator[](i), pNneighbors, i, pRawData);
                    } else {
                        exactNeighbors = 
                            mOriginalData->euclidianDistance(neighborhood_->neighbors->operator[](i), pNneighbors, i, pRawData);
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

distributionInverseIndex* MinHash::getDistributionOfInverseIndex() {
    return mInverseIndex->getDistribution();
}
