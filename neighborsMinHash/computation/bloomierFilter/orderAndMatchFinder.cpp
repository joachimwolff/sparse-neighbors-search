#include <unordered_map>
#include <cmath>
#include "orderAndMatchFinder.h"

#ifdef OPENMP
#include <omp.h>
#endif
OrderAndMatchFinder::OrderAndMatchFinder(const size_t pModulo, const size_t pNumberOfElements, BloomierHash* pBloomierHash) {
    mModulo = pModulo;
    mNumberOfElements = pNumberOfElements;
    mPiVector = new vsize_t();
    mTauVector = new vsize_t();
    mBloomierHash = pBloomierHash;
    const size_t sizeOfBloomFilter = ceil(mModulo/ 8.0);
    mBloomFilterHashesSeen = new bitVector(sizeOfBloomFilter);
    mBloomFilterNonSingeltons = new bitVector(sizeOfBloomFilter);
    mBloomFilterInstance = new bitVector(sizeOfBloomFilter);
    mBloomFilterInstanceDifferentSeed = new bitVector(sizeOfBloomFilter);
    mSeeds = new std::unordered_map<size_t, size_t>;
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
size_t OrderAndMatchFinder::deleteValueInBloomFilterInstance(const size_t pKey) {
    const unsigned char index = floor(pKey / 8.0);
    const unsigned char value = 1 << (pKey % 8);
    (*mBloomFilterInstance)[index] = (*mBloomFilterInstance)[index] ^ value;
}
size_t OrderAndMatchFinder::getSeed(const size_t pKey) const{
    const unsigned char index = floor(pKey / 8.0);
    const unsigned char value = 1 << (pKey % 8);
    const unsigned char valueSeenBefor = (*mBloomFilterInstance)[index] & value;
    if (valueSeenBefor == value) {
        // value seen and not using default seed
        const unsigned char differentSeed = (*mBloomFilterInstanceDifferentSeed)[index] & value;
        if (differentSeed == value) {
            return (*mSeeds)[pKey];
        }
        // value seen using default seed
        return MAX_VALUE - 1;
    }
    // value never seen
    return MAX_VALUE;
}
void OrderAndMatchFinder::findMatch(const size_t pKey, vsize_t* pNeighbors) {
    const int singeltonValue = this->tweak(pKey, pNeighbors);
    this->computeNonSingeltons(pNeighbors);
    const unsigned char index = floor(pKey / 8.0);
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
        // std::cout << "tweak i:" << i << std::endl;
        mBloomierHash->getKNeighbors(pKey, seed, pNeighbors);
        j = 0;
        bool breakForOpenMp = true;
        for (auto it = pNeighbors->begin(); it != pNeighbors->end() && breakForOpenMp; ++it) {
            
            index = floor((*it) / 8.0);
            value = 1 << ((*it) % 8);
#pragma omp critical
            {
                valueSeen = (*mBloomFilterHashesSeen)[index] & value;
                if (value != valueSeen) {
    
                    (*mBloomFilterNonSingeltons)[index] = (*mBloomFilterNonSingeltons)[index] | value;
                    if (mBloomierHash->getHashSeed() != seed) {
                        index = floor(pKey / 8.0);
                        value = 1 << (pKey % 8);
    
                        {
                            (*mBloomFilterInstanceDifferentSeed)[index] = (*mBloomFilterInstanceDifferentSeed)[index] | value;
                            (*mSeeds)[pKey] = seed;
                        }
                    }
                    singelton = j;
                    breakForOpenMp = false;
                    // it = pNeighbors->end();
                    // break; 
                }
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
    // std::cout << "size if neighbors: " << pNeighbors->size() << std::endl;
    for (auto itNeighbors = pNeighbors->begin(); itNeighbors != pNeighbors->end(); ++itNeighbors) {
        index = floor((*itNeighbors) / 8.0);
        value = 1 << ((*itNeighbors) % 8);
        valueSeen = (*mBloomFilterHashesSeen)[index] & value;
        if (valueSeen == value) {
            (*mBloomFilterNonSingeltons)[index] = (*mBloomFilterNonSingeltons)[index] | value;
        }
    }
    // for (auto itNeighbors = pNeighbors->begin(); itNeighbors != pNeighbors->end(); ++itNeighbors){
    //     index = floor((*itNeighbors) / 8.0);
    //     value = 1 << ((*itNeighbors) % 8);
    //     (*mBloomFilterHashesSeen)[index] = (*mBloomFilterHashesSeen)[index] | value;
    // }
}