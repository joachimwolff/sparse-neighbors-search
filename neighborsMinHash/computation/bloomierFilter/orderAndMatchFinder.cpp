#include <unordered_map>
#include <cmath>
#include "orderAndMatchFinder.h"

OrderAndMatchFinder::OrderAndMatchFinder(size_t pModulo, size_t pNumberOfElements, BloomierHash* pBloomierHash) {
    mModulo = pModulo;
    mNumberOfElements = pNumberOfElements;
    mPiVector = new vsize_t();
    mTauVector = new vsize_t();
    mBloomierHash = pBloomierHash;
    size_t sizeOfBloomFilter = ceil(mModulo/ 8.0);
    mBloomFilterHashesSeen = new bitVector(sizeOfBloomFilter);
    mBloomFilterNonSingeltons = new bitVector(sizeOfBloomFilter);
    mBloomFilterInstance = new bitVector(sizeOfBloomFilter);
    mBloomFilterInstanceDifferentSeed = new bitVector(sizeOfBloomFilter);
    mSeeds = new std::unordered_map<size_t, size_t>;
    mBloomFilterSeed = 42;
    mHash = new Hash();
}
OrderAndMatchFinder::~OrderAndMatchFinder() {
    delete mPiVector;
    delete mTauVector;
    delete mSeeds;
    delete mHash;
    delete mBloomFilterHashesSeen;
    delete mBloomFilterNonSingeltons;
    delete mBloomFilterInstance;
    delete mBloomFilterInstanceDifferentSeed;
}

size_t OrderAndMatchFinder::getSeed(size_t pKey) {
    unsigned char index = floor(pKey / 8.0);
    unsigned char value = 1 << (pKey % 8);
    unsigned char valueSeenBefor = (*mBloomFilterInstance)[index] & value;
    if (valueSeenBefor == value) {
        // value seen and not using default seed
        unsigned char differentSeed = (*mBloomFilterInstanceDifferentSeed)[index] & value;
        if (differentSeed == value) {
            return (*mSeeds)[pKey];
        }
        // value seen but default seed
        return MAX_VALUE - 1;
    }
    // value never seen but default seed
    return MAX_VALUE;

}
void OrderAndMatchFinder::findMatch(size_t pKey, vsize_t* pNeighbors) {
    
    int singeltonValue = this->tweak(pKey, mBloomierHash->getHashSeed(), pNeighbors);
    this->computeNonSingeltons(pNeighbors);
    unsigned char index = floor(pKey / 8.0);
    unsigned char value = 1 << (pKey % 8);
    (*mBloomFilterInstance)[index] = (*mBloomFilterInstance)[index] | value;
    mPiVector->push_back(pKey);
    mTauVector->push_back(singeltonValue);
}

vsize_t* OrderAndMatchFinder::findIndexAndReturnNeighborhood(size_t key) {
    vsize_t* neighbors = new vsize_t(mNumberOfElements);
    this->findMatch(key, neighbors);
    return neighbors;
    
}
vsize_t* OrderAndMatchFinder::getPiVector() {
    return mPiVector;
}
vsize_t* OrderAndMatchFinder::getTauVector() {
    return mTauVector;
}
int OrderAndMatchFinder::tweak(size_t pKey, size_t pSeed, vsize_t* pNeighbors) {
    int singelton = -1;
    size_t i = 0;
    size_t j = 0;
    unsigned char value = 0;
    unsigned char valueSeen = 0;
    unsigned char index = 0;
    
    while (singelton == -1) {
        std::cout << "tweak i:" << i << std::endl;
        pSeed = pSeed+i;
        mBloomierHash->getKNeighbors(pKey, pSeed, pNeighbors);
        j = 0;
        for (auto it = pNeighbors->begin(); it != pNeighbors->end(); ++it) {
            
            index = floor((*it) / 8.0);
            value = 1 << ((*it) % 8);
            valueSeen = (*mBloomFilterHashesSeen)[index] & value;
            // std::cout << 
            if (value != valueSeen) {
                (*mBloomFilterNonSingeltons)[index] = (*mBloomFilterNonSingeltons)[index] | value;
                if (mBloomierHash->getHashSeed() != pSeed) {
                    index = floor(pKey / 8.0);
                    value = 1 << (pKey % 8);
                    (*mBloomFilterInstanceDifferentSeed)[index] = (*mBloomFilterInstanceDifferentSeed)[index] | value;
                    (*mSeeds)[pKey] = pSeed;
                }
                singelton = j;
                break;
            }
            ++j;
        }
        ++i;
    }
    return singelton;
}

void OrderAndMatchFinder::computeNonSingeltons(vsize_t* pNeighbors) {
    unsigned char value;
    unsigned char valueSeen;
    unsigned char index;
    for (auto itNeighbors = pNeighbors->begin(); itNeighbors != pNeighbors->end(); ++itNeighbors) {
        index = floor((*itNeighbors) / 8.0);
        value = 1 << ((*itNeighbors) % 8);
        valueSeen = (*mBloomFilterHashesSeen)[index] & value;
        if (valueSeen != value) {
            (*mBloomFilterNonSingeltons)[index] = (*mBloomFilterNonSingeltons)[index] | value;
        }
    }
    for (auto itNeighbors = pNeighbors->begin(); itNeighbors != pNeighbors->end(); ++itNeighbors){
        index = floor((*itNeighbors) / 8.0);
        value = 1 << ((*itNeighbors) % 8);
        (*mBloomFilterHashesSeen)[index] = (*mBloomFilterHashesSeen)[index] | value;
    }
}