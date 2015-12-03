#include <unordered_map>
#include "orderAndMatchFinder.h"

OrderAndMatchFinder::OrderAndMatchFinder(size_t pModulo, size_t pNumberOfElements, BloomierHash* pBloomierHash) {
    mModulo = pModulo;
    mNumberOfElements = pNumberOfElements;
    // mQ = pQ;
    mPiVector = new vsize_t();
    mTauVector = new vsize_t();
    mBloomierHash = pBloomierHash;
    mHashesSeen = new std::set<size_t>();
    mNonSingeltons = new std::set<size_t>();
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
         if (this->findMatch(subsetNextRecursion) == false) {
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
    size_t i = 0;
    while (!(this->findMatch(pSubset)) && i < 5) ++i;
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
    vsize_t*  neighbors = mBloomierHash->getKNeighbors(pKey);
    for (auto it = neighbors->begin(); it != neighbors->end(); ++it) {
        if (mNonSingeltons->find((*it)) == mNonSingeltons->end()) {
            delete neighbors;
            return i;
        }
        ++i;
    } 
    delete neighbors;   
    return -1;
}

void OrderAndMatchFinder::computeNonSingeltons(vsize_t* pKeyValues) {
    for (auto it = pKeyValues->begin(); it != pKeyValues->end(); ++it) {
        vsize_t* neighbors = mBloomierHash->getKNeighbors((*it));
        for (auto itNeighbors = neighbors->begin(); itNeighbors != neighbors->end(); ++itNeighbors){
            if (mHashesSeen->find((*itNeighbors)) != mHashesSeen->end()) {
                mNonSingeltons->insert((*itNeighbors));
            }
        }
        for (auto itNeighbors = neighbors->begin(); itNeighbors != neighbors->end(); ++itNeighbors){
            mHashesSeen->insert((*itNeighbors));
        }
        delete neighbors;
    } 
}