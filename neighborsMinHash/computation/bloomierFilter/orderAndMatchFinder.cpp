#include <cmath>
#include "orderAndMatchFinder.h"

OrderAndMatchFinder::OrderAndMatchFinder(const size_t pModulo, const size_t pNumberOfElements, BloomierHash* pBloomierHash) {
    mModulo = pModulo;
    mNumberOfElements = pNumberOfElements;
    mTauVector = new vsize_t();
    mBloomierHash = pBloomierHash; 
    mDifferentSeed = new std::set<size_t>();
    mBloomFilterSeed = 42;
}
OrderAndMatchFinder::~OrderAndMatchFinder() {
    delete mTauVector;
}
size_t OrderAndMatchFinder::getSeed(const size_t pKey) const {
    auto seed = mDifferentSeed->find(pKey);
    if (seed != mDifferentSeed->end()) return *seed;
    return mBloomierHash->getHashSeed();
}
void OrderAndMatchFinder::findMatch(const size_t pKey, vsize_t* pNeighbors, const vvsize_t_p* pValueTable) {
    const int singeltonValue = this->tweak(pKey, pNeighbors, pValueTable);
    mTauVector->push_back(singeltonValue);
}

vsize_t* OrderAndMatchFinder::findIndexAndReturnNeighborhood(const size_t key, const vvsize_t_p* pValueTable) {
    vsize_t* neighbors = new vsize_t(mNumberOfElements);
    this->findMatch(key, neighbors, pValueTable);
    return neighbors;
}
vsize_t* OrderAndMatchFinder::getTauVector() const {
    return mTauVector;
}
int OrderAndMatchFinder::tweak(const size_t pKey, vsize_t* pNeighbors, const vvsize_t_p* pValueTable) {
    int singelton = -1;
    size_t j = 0;
    // size_t index = 0;
    size_t seed = mBloomierHash->getHashSeed();
    while (singelton == -1) {
        mBloomierHash->getKNeighbors(pKey, seed, pNeighbors);
        j = 0;
        bool breakForOpenMp = true;
        for (auto it = pNeighbors->begin(); it != pNeighbors->end() && breakForOpenMp; ++it) {
            if ((*pValueTable)[pNeighbors->at(j)] == NULL) {
                singelton = j;
                breakForOpenMp = false;
                if (seed != mBloomierHash->getHashSeed()) {
                    mDifferentSeed->insert(pKey);
                }
            }
            ++j;
        }
        ++seed;     
        if (seed == mBloomierHash->getHashSeed()+20) {
            bool allValuesUsed = false;
            for (size_t i = 0; i < pValueTable->size(); ++i) {
                if ((*pValueTable)[i] == NULL) {
                    allValuesUsed = true;
                    break;
                }
            }
            if (allValuesUsed){
                std::cout << "All places are used!" << std::endl;
            } 
            else { 
                std::cout << "NOT all places are used, but do not find a free one!" << std::endl;
            }
            return -1;
        }   
    }
    std::cout << "Singelton found: " << pNeighbors->at(singelton) << std::endl;
    return singelton;
}