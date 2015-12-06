#include <unordered_map>
#include "orderAndMatchFinder.h"

OrderAndMatchFinder::OrderAndMatchFinder(size_t pModulo, size_t pNumberOfElements, vsize_t* pSubset, BloomierHash* pBloomierHash) {
    mModulo = pModulo;
    mNumberOfElements = pNumberOfElements;
    // mQ = pQ;
    mPiVector = new vsize_t();
    mTauVector = new vsize_t();
    mBloomierHash = pBloomierHash;
    mHashesSeen = new std::set<size_t>();
    mNonSingeltons = new std::set<size_t>();
    this->computeNonSingeltons(pSubset);
    mSeeds = new std::unordered_map<size_t, size_t>;
}
OrderAndMatchFinder::~OrderAndMatchFinder() {
    delete mPiVector;
    delete mTauVector;
    delete mSeeds;
}

size_t OrderAndMatchFinder::getSeed(size_t pKey) {
    return (*mSeeds)[pKey];
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
        // std::cout << "instance: " << (*pSubset)[i] << " singeltonValue: " << singeltonValue << std::endl;
        if (singeltonValue != -1) {
            piVector->push_back((*pSubset)[i]);
            tauVector->push_back(singeltonValue);
        } // else {
        //     // std::cout << "tweak == -1: " << (*pSubset)[i] << std::endl;
        //     subsetNextRecursion->push_back((*pSubset)[i]);
        // }
    }
    if (piVector->size() == 0) {
        delete piVector;
        delete tauVector;
        delete subsetNextRecursion;
        return false;
    }
    // if (subsetNextRecursion->size() != 0) {
    //      if (this->findMatch(subsetNextRecursion) == false) {
    //         delete piVector;
    //         delete tauVector;
    //         delete subsetNextRecursion;
    //         return false;
    //      }
    // }
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
    // this->findMatch(pSubset);
    while (!(this->findMatch(pSubset)) && i < 100) {
        ++i;
        delete mPiVector;
        delete mTauVector;
        delete mHashesSeen;
        delete mNonSingeltons;
        mPiVector = new vsize_t();
        mTauVector = new vsize_t();
        // mBloomierHash = pBloomierHash;
        mHashesSeen = new std::set<size_t>();
        mNonSingeltons = new std::set<size_t>();
        this->computeNonSingeltons(pSubset);
        mBloomierHash->setHashSeed(mBloomierHash->getHashSeed() + i);
    } 
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
        // std::cout << "instance: " << pKey << " hashSeed: " << pSeed << "\tneighbors: ";
        
        size_t j = 0;
        for (auto it = neighbors->begin(); it != neighbors->end(); ++it) {
                // std::cout << *it << " ";
            // std::cout << "it: " << *it << std::endl;
            
            if (mNonSingeltons->find((*it)) == mNonSingeltons->end()) {
                // mHashesSeen->insert((*it));
                // mNonSingeltons->insert(*it);
                // std::cout << "Singeltin: "<< *it << std::endl;
                mNonSingeltons->insert((*it));
                
                delete neighbors;
            // std::cout << std::endl;
                (*mSeeds)[pKey] = pSeed;
                return j;
            }
            ++j;
        }
        // std::cout << "No singelton found." << std::endl;
        delete neighbors;
        ++i;
    }
    
    
}

void OrderAndMatchFinder::computeNonSingeltons(vsize_t* pKeyValues, size_t pSeed) {
    for (auto it = pKeyValues->begin(); it != pKeyValues->end(); ++it) {
        vsize_t* neighbors = mBloomierHash->getKNeighbors((*it), pSeed);
        // std::cout << "instance: " << *it << " hashSeed: " << mBloomierHash->getHashSeed() << "\tneighbors: ";
        for (auto itNeighbors = neighbors->begin(); itNeighbors != neighbors->end(); ++itNeighbors){
            // std::cout << *itNeighbors << " ";
            if (mHashesSeen->find((*itNeighbors)) != mHashesSeen->end()) {
                mNonSingeltons->insert((*itNeighbors));
            }
        }
        // std::cout << std::endl;
        // for (auto itNeighbors = neighbors->begin(); itNeighbors != neighbors->end(); ++itNeighbors){
        //     mHashesSeen->insert((*itNeighbors));
        // }
        delete neighbors;
    } 
}