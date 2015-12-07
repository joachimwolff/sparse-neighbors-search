#include <unordered_map>
#include "orderAndMatchFinder.h"

OrderAndMatchFinder::OrderAndMatchFinder(size_t pModulo, size_t pNumberOfElements, BloomierHash* pBloomierHash) {
    mModulo = pModulo;
    mNumberOfElements = pNumberOfElements;
    // mQ = pQ;
    mPiVector = new vsize_t();
    mTauVector = new vsize_t();
    mBloomierHash = pBloomierHash;
    mBloomFilterHashesSeen = 0;
    mBloomFilterNonSingeltons = 0;
    // this->computeNonSingeltons(pSubset);
    mSeeds = new std::unordered_map<size_t, size_t>;
    mBloomFilterInstance = 0;
    mBloomFilterInstanceDifferentSeed = 0;
    mBloomFilterSeed = 42;

    mHash = new Hash();
}
OrderAndMatchFinder::~OrderAndMatchFinder() {
    delete mPiVector;
    delete mTauVector;
    delete mSeeds;
}

size_t OrderAndMatchFinder::getSeed(size_t pKey) {
    
    size_t value = mHash->hash(pKey+1, mBloomFilterSeed, MAX_VALUE);
    size_t valueSeenBefor = mBloomFilterInstance & value;
    if (valueSeenBefor == value) {
        // value seen and not using default seed
        size_t differentSeed = mBloomFilterInstanceDifferentSeed & value;
        if (differentSeed == value) {
            return (*mSeeds)[pKey];
        }
        // value seen but default seed
        return MAX_VALUE - 1;
    }
    // value never seen but default seed
    return MAX_VALUE;

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
        singeltonValue = tweak((*pSubset)[i], pSubset, mBloomierHash->getHashSeed());
        mBloomFilterInstance = mBloomFilterInstance | mHash->hash((*pSubset)[i]+1, mBloomFilterSeed, MAX_VALUE);
        if (singeltonValue != -1) {
            piVector->push_back((*pSubset)[i]);
            tauVector->push_back(singeltonValue);
        }
    }
    if (piVector->size() == 0) {
        delete piVector;
        delete tauVector;
        delete subsetNextRecursion;
        return false;
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
    this->computeNonSingeltons(pSubset);
    this->findMatch(pSubset);
}
vsize_t* OrderAndMatchFinder::getPiVector() {
    return mPiVector;
}
vsize_t* OrderAndMatchFinder::getTauVector() {
    return mTauVector;
}
int OrderAndMatchFinder::tweak (size_t pKey, vsize_t* pSubset, size_t pSeed) {
    
    int singelton = -1;
    size_t i = 0;
    while (singelton == -1) {
        pSeed = pSeed+i;
        vsize_t*  neighbors = mBloomierHash->getKNeighbors(pKey, pSeed);
        size_t j = 0;
        for (auto it = neighbors->begin(); it != neighbors->end(); ++it) {
            size_t value = mHash->hash((*it) + 1, mBloomFilterSeed, MAX_VALUE);
            size_t valueSeen = mBloomFilterNonSingeltons & value;
            if (value != valueSeen) {
                mBloomFilterNonSingeltons = mBloomFilterNonSingeltons | value;
                delete neighbors;
                if (mBloomierHash->getHashSeed() != pSeed) {
                    mBloomFilterInstanceDifferentSeed = mBloomFilterInstanceDifferentSeed | mHash->hash(pKey + 1, mBloomFilterSeed, MAX_VALUE);
                    (*mSeeds)[pKey] = pSeed;
                }
                return j;
            }
            ++j;
        }
        delete neighbors;
        ++i;
    }
    return -1;
}

void OrderAndMatchFinder::computeNonSingeltons(vsize_t* pKeyValues, size_t pSeed) {
    for (auto it = pKeyValues->begin(); it != pKeyValues->end(); ++it) {
        vsize_t* neighbors = mBloomierHash->getKNeighbors((*it), pSeed);
        for (auto itNeighbors = neighbors->begin(); itNeighbors != neighbors->end(); ++itNeighbors) {
            size_t value = mHash->hash((*itNeighbors) + 1, mBloomFilterSeed, MAX_VALUE);
            size_t valueSeen = mBloomFilterHashesSeen & value;
            if (valueSeen != value) {
                mBloomFilterNonSingeltons = mBloomFilterNonSingeltons & value;
            }
            // if (mHashesSeen->find((*itNeighbors)) != mHashesSeen->end()) {
            //     mNonSingeltons->insert((*itNeighbors));
            // }
        }
        for (auto itNeighbors = neighbors->begin(); itNeighbors != neighbors->end(); ++itNeighbors){
            mBloomFilterHashesSeen = mBloomFilterHashesSeen | mHash->hash((*itNeighbors) + 1, mBloomFilterSeed, MAX_VALUE);
        }
        delete neighbors;
    } 
}