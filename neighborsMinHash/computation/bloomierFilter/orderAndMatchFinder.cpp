#include <cmath>
#include "orderAndMatchFinder.h"

OrderAndMatchFinder::OrderAndMatchFinder(const size_t pModulo, const size_t pNumberOfElements, BloomierHash* pBloomierHash) {
    mModulo = pModulo;
    mNumberOfElements = pNumberOfElements;
    // mPiVector = new vsize_t();
    mTauVector = new vsize_t();
    mBloomierHash = pBloomierHash; 
    mSizeOfBloomFilter = ceil(mModulo/ 8.0);
    mBloomFilterHashesSeen = new bitVector[mSizeOfBloomFilter];
    // mBloomFilterNonSingeltons = new bitVector[mSizeOfBloomFilter];
    // mBloomFilterInstance = new bitVector[mSizeOfBloomFilter];
    // mBloomFilterInstanceDifferentSeed = new bitVector[mSizeOfBloomFilter];
    for (size_t i = 0; i < mSizeOfBloomFilter; ++i) {
        mBloomFilterHashesSeen[i] = 0;
        // mBloomFilterNonSingeltons[i] = 0;
        // mBloomFilterInstance[i] = 0;
        // mBloomFilterInstanceDifferentSeed[i] = 0;
    }
    mBloomFilterSeed = 42;
    // mSeeds = new std::unordered_map<size_t, size_t>();
}
OrderAndMatchFinder::~OrderAndMatchFinder() {
    // delete mPiVector;
    delete mTauVector;
    // delete mSeeds;
    delete [] mBloomFilterHashesSeen;
    // delete [] mBloomFilterNonSingeltons;
    // delete [] mBloomFilterInstance;
    // delete [] mBloomFilterInstanceDifferentSeed;
}
// void OrderAndMatchFinder::deleteValueInBloomFilterInstance(const size_t pKey) {
//     // const size_t index = floor(pKey / 8);
//     // const unsigned char value = 1 << (pKey % 8);
//     // mBloomFilterInstance[index] = mBloomFilterInstance[index] ^ value;
// }
// bool OrderAndMatchFinder::getValueSeenBefor(const size_t pKey) const {
//     // const size_t index = floor(pKey / 8);
//     // const unsigned char value = 1 << (pKey % 8);
//     // const unsigned char valueSeenBefor = mBloomFilterInstance[index] & value;
//     // if (valueSeenBefor == value) return true;
//     return false;
// }
// size_t OrderAndMatchFinder::getSeed(const size_t pKey) const {
//     // const size_t index = pKey / 8;
//     // const unsigned char value = 1 << (pKey % 8);
//     // const unsigned char valueSeenBefor = mBloomFilterInstanceDifferentSeed[index] & value;
//     // if (valueSeenBefor == value) return (*mSeeds)[pKey];
//     // return mBloomierHash->getHashSeed();
// }
void OrderAndMatchFinder::findMatch(const size_t pKey, vsize_t* pNeighbors) {
    // std::cout << "Searching singelton..." << std::endl;
    const int singeltonValue = this->tweak(pKey, pNeighbors);
    // this->computeNonSingeltons(pNeighbors);
    // const size_t index = pKey / 8;
    // const unsigned char value = 1 << (pKey % 8);
    // mBloomFilterInstance[index] = mBloomFilterInstance[index] | value;
    // mPiVector->push_back(pKey);
    mTauVector->push_back(singeltonValue);
}

vsize_t* OrderAndMatchFinder::findIndexAndReturnNeighborhood(const size_t key) {
    vsize_t* neighbors = new vsize_t(mNumberOfElements);
    this->findMatch(key, neighbors);
    return neighbors;
}
// vsize_t* OrderAndMatchFinder::getPiVector() const {
//     return mPiVector;
// }
vsize_t* OrderAndMatchFinder::getTauVector() const {
    return mTauVector;
}
int OrderAndMatchFinder::tweak(const size_t pKey, vsize_t* pNeighbors) {
    int singelton = -1;
    size_t j = 0;
    unsigned char value = 0;
    unsigned char valueSeen = 0;
    size_t index = 0;
    size_t seed = mBloomierHash->getHashSeed();
    while (singelton == -1) {
        mBloomierHash->getKNeighbors(pKey, seed, pNeighbors);
        j = 0;
        bool breakForOpenMp = true;
        for (auto it = pNeighbors->begin(); it != pNeighbors->end() && breakForOpenMp; ++it) {
            
            index = (*it) / 8;
            value = 1 << ((*it) % 8);
            valueSeen = mBloomFilterHashesSeen[index] & value;
            
            if (value != valueSeen) {
                // mBloomFilterNonSingeltons[index] = mBloomFilterNonSingeltons[index] | value;
                singelton = j;
                breakForOpenMp = false;
                // if (seed != mBloomierHash->getHashSeed()) {
                    // (*mSeeds)[pKey] = seed;
                    // index = pKey / 8;
                    // value = 1 << (pKey % 8);
                    // mBloomFilterInstanceDifferentSeed[index] = mBloomFilterInstanceDifferentSeed[index] | value;
                // }
            }
            ++j;
        }
        ++seed;        
    }
    // std::cout << "Singeltin found: " << pNeighbors->at(singelton) << std::endl;
    for (auto it = pNeighbors->begin(); it != pNeighbors->end(); ++it) {
         index = (*it) / 8;
        //  std::cout << "it: " << (*it) << " size of bloomier filter: " << mSizeOfBloomFilter << std::endl;
        //  std::cout << "index: " << static_cast<size_t> (index) << std::endl;
         value = 1 << ((*it) % 8);
         mBloomFilterHashesSeen[index] = mBloomFilterHashesSeen[index] | value;
    }
     
    return singelton;
}

// void OrderAndMatchFinder::computeNonSingeltons(const vsize_t* pNeighbors) {
//     unsigned char value;
//     unsigned char valueSeen;
//     unsigned char index;
//     for (auto itNeighbors = pNeighbors->begin(); itNeighbors != pNeighbors->end(); ++itNeighbors) {
//         index = (*itNeighbors) / mSizeOfBloomFilter;
//         value = 1 << ((*itNeighbors) % 8);
//         valueSeen = mBloomFilterHashesSeen[index] & value;
//         if (valueSeen == value) {
//             mBloomFilterNonSingeltons[index] = mBloomFilterNonSingeltons[index] | value;
//         }
//     }
//     for (auto itNeighbors = pNeighbors->begin(); itNeighbors != pNeighbors->end(); ++itNeighbors){
//         index = (*itNeighbors) / mSizeOfBloomFilter;
//         value = 1 << ((*itNeighbors) % 8);
//         (*mBloomFilterHashesSeen)[index] = (*mBloomFilterHashesSeen)[index] | value;
//     }
// }