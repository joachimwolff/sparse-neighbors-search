#include <cmath>
#include "orderAndMatchFinder.h"

OrderAndMatchFinder::OrderAndMatchFinder(const size_t pModulo, const size_t pNumberOfElements, BloomierHash* pBloomierHash) {
    mModulo = pModulo;
    mNumberOfElements = pNumberOfElements;
    mPiVector = new vsize_t();
    mTauVector = new vsize_t();
    mBloomierHash = pBloomierHash; 
    mSizeOfBloomFilter = ceil(mModulo/ 8.0);
    mBloomFilterHashesSeen = new bitVector(mSizeOfBloomFilter);
    mBloomFilterNonSingeltons = new bitVector(mSizeOfBloomFilter);
    mBloomFilterInstance = new bitVector(mSizeOfBloomFilter);
    mBloomFilterSeed = 42;
}
OrderAndMatchFinder::~OrderAndMatchFinder() {
    delete mPiVector;
    delete mTauVector;
    delete mSeeds;
    delete mBloomFilterHashesSeen;
    delete mBloomFilterNonSingeltons;
    delete mBloomFilterInstance;
    delete mBloomFilterInstanceDifferentSeed;
}
void OrderAndMatchFinder::deleteValueInBloomFilterInstance(const size_t pKey) {
    const unsigned char index = pKey / mSizeOfBloomFilter;
    const unsigned char value = 1 << (pKey % 8);
    (*mBloomFilterInstance)[index] = (*mBloomFilterInstance)[index] ^ value;
}
bool OrderAndMatchFinder::getValueSeenBefor(const size_t pKey) const{
    const unsigned char index = pKey / mSizeOfBloomFilter;
    const unsigned char value = 1 << (pKey % 8);
    const unsigned char valueSeenBefor = (*mBloomFilterInstance)[index] & value;
    if (valueSeenBefor == value) return true;
    return false;
}
void OrderAndMatchFinder::findMatch(const size_t pKey, vsize_t* pNeighbors) {
    const int singeltonValue = this->tweak(pKey, pNeighbors);
    this->computeNonSingeltons(pNeighbors);
    const unsigned char index = pKey / mSizeOfBloomFilter;
    const unsigned char value = 1 << (pKey % 8);
    (*mBloomFilterInstance)[index] = (*mBloomFilterInstance)[index] | value;
    mPiVector->push_back(pKey);
    mTauVector->push_back(singeltonValue);
}

vsize_t* OrderAndMatchFinder::findIndexAndReturnNeighborhood(const size_t key) {
    vsize_t* neighbors = new vsize_t(mNumberOfElements);
    this->findMatch(key, neighbors);
    return neighbors;
}
vsize_t* OrderAndMatchFinder::getPiVector() const {
    return mPiVector;
}
vsize_t* OrderAndMatchFinder::getTauVector() const {
    return mTauVector;
}
int OrderAndMatchFinder::tweak(const size_t pKey, vsize_t* pNeighbors) {
    int singelton = -1;
    size_t j = 0;
    unsigned char value = 0;
    unsigned char valueSeen = 0;
    unsigned char index = 0;
    size_t seed = mBloomierHash->getHashSeed();
    while (singelton == -1) {
        mBloomierHash->getKNeighbors(pKey, seed, pNeighbors);
        j = 0;
        bool breakForOpenMp = true;
        for (auto it = pNeighbors->begin(); it != pNeighbors->end() && breakForOpenMp; ++it) {
            
            index = (*it) / mSizeOfBloomFilter;
            value = 1 << ((*it) % 8);
                valueSeen = (*mBloomFilterHashesSeen)[index] & value;
                if (value != valueSeen) {
    
                    (*mBloomFilterNonSingeltons)[index] = (*mBloomFilterNonSingeltons)[index] | value;
                    singelton = j;
                    breakForOpenMp = false;
            }
            ++j;
        }
        ++seed;        
    }
    return singelton;
}

void OrderAndMatchFinder::computeNonSingeltons(const vsize_t* pNeighbors) {
    unsigned char value;
    unsigned char valueSeen;
    unsigned char index;
    for (auto itNeighbors = pNeighbors->begin(); itNeighbors != pNeighbors->end(); ++itNeighbors) {
        index = (*itNeighbors) / mSizeOfBloomFilter;
        value = 1 << ((*itNeighbors) % 8);
        valueSeen = (*mBloomFilterHashesSeen)[index] & value;
        if (valueSeen == value) {
            (*mBloomFilterNonSingeltons)[index] = (*mBloomFilterNonSingeltons)[index] | value;
        }
    }
    // for (auto itNeighbors = pNeighbors->begin(); itNeighbors != pNeighbors->end(); ++itNeighbors){
    //     index = (*itNeighbors) / mSizeOfBloomFilter;
    //     value = 1 << ((*itNeighbors) % 8);
    //     (*mBloomFilterHashesSeen)[index] = (*mBloomFilterHashesSeen)[index] | value;
    // }
}