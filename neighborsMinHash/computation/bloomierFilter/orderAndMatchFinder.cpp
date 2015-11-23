#include <unordered_map>
// #include "orderAndMatchFinder.h"

OrderAndMatchFinder::OrderAndMatchFinder(size_t pM, size_t pK, size_t pQ, BloomierHash* pBloomierHash) {
    mM = pM;
    mK = pK;
    mQ = pQ;
    mPiVector = new vsize_t();
    mTauVector = new vsize_t();
    mBloomierHash = pBloomierHash;
}
OrderAndMatchFinder::~OrderAndMatchFinder() {
    delete mPiVector;
    delete mTauVector;
}

bool OrderAndMatchFinder::findMatch(vsize_t* pSubset) {
    if (pSubset->size() == 0) {
        return true;
    }
    vsize_t* piVector = new vsize_t();
    // vsize_t* orderingVector = new vsize_t();
    vsize_t* tauVector = new vsize_t();
    vsize_t* subsetNextRecursion = new vsize_t();
    int singeltonValue;
    for (size_t i = 0; i < pSubset->size(); ++i) {
        
        singeltonValue = tweak((*pSubset)[i], pSubset);
        if (singeltonValue != -1) {
            piVector->push_back((*pSubset)[i]);
            tauVector->push_back(singeltonValue);
        } else {
            subsetNextRecursion->push_back((*pSubset)[i]);
        }
    }
    if (piVector->size() == 0) {
        return false;
    }
    if (subsetNextRecursion->size() != 0) {
         if (this->find(subsetNextRecursion) == false) {
             return false;
         }
    }
    for (size_t i = 0; i < piVector->size(); ++i) {
        mPiVector->push_back((*piVector)[i]);
    }
    for (size_t i = 0; i < tauVector->size(); ++i) {
        mTauVector->push_back((*tauVector)[i]);
    }
    delete piVector;
    delete tauVector;
    delete subsetNextRecursion;
    return true;
}

void OrderAndMatchFinder::find(vsize_t* pSubset) {
    this->findMatch(pSubset);
}
vsize_t* OrderAndMatchFinder::getPiVector() {
    return mPiVector;
}
vsize_t* OrderAndMatchFinder::getTauVector() {
    return mTauVector;
}
int OrderAndMatchFinder::tweak (size_t pKey, vsize_t* pSubset) {
    size_t i = 0;
    this->computeNonSingeltons(pSubset);
    vsize_t*  neighbors = mBloomierHash->getKNeighbors(pKey, mK, mM);
    for (auto it = neighbors->begin(); it != neighbors->end(); ++it) {
        if (mNonSingeltons->find((*it)) == mNonSingeltons->end()) {
            return i;
        }
        ++i;
    } 
    return -1;
}

void OrderAndMatchFinder::computeNonSingeltons(vsize_t* pKeyValues) {
    for (auto it = pKeyValues->begin(); it != pKeyValues->end(); ++i) {
        vsize_t* neighbors = mBloomierHash->getKNeighbors((*it), mK, mM);
        for (auto itNeighbors = neighbors->begin(); itNeighbors != neighbors->end(); ++itNeighbors){
            if (mHashesSeen->find((*itNeighbors)) != mHashesSeen->end()) {
                mNonSingeltons->push_back((*itNeighbors));
            }
        }
        for (auto itNeighbors = neighbors->begin(); itNeighbors != neighbors->end(); ++itNeighbors){
            mHashesSeen->push_back((*itNeighbors));
        }
    }
}