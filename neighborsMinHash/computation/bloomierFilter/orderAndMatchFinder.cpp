#include <unordered_map>
#include "orderAndMatchFinder.h"

OrderAndMatchFinder::OrderAndMatchFinder(size_t pM, size_t pK, size_t pQ, BloomierHash* pBloomierHash) {
    mM = pM;
    mK = pK;
    mQ = pQ;
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
    	// std::cout << "20" << std::endl;

    if (pSubset->size() == 0) {
        return true;
    }
    	// std::cout << "25" << std::endl;

    vsize_t* piVector = new vsize_t();
    // vsize_t* orderingVector = new vsize_t();
    vsize_t* tauVector = new vsize_t();
    vsize_t* subsetNextRecursion = new vsize_t();
    int singeltonValue;
    for (size_t i = 0; i < pSubset->size(); ++i) {
        	// std::cout << "33" << std::endl;

        singeltonValue = tweak((*pSubset)[i], pSubset);
        	// std::cout << "36" << std::endl;

        if (singeltonValue != -1) {
            	// std::cout << "39" << std::endl;

            piVector->push_back((*pSubset)[i]);
            	// std::cout << "42" << std::endl;

            tauVector->push_back(singeltonValue);
            	// std::cout << "45" << std::endl;

        } else {
            	// std::cout << "48" << std::endl;

            subsetNextRecursion->push_back((*pSubset)[i]);
            	// std::cout << "51" << std::endl;

        }
    }
    	// std::cout << "55" << std::endl;

    if (piVector->size() == 0) {
        return false;
    }
    	// std::cout << "60" << std::endl;

    if (subsetNextRecursion->size() != 0) {
         if (this->findMatch(subsetNextRecursion) == false) {
             return false;
         }
    }
    	// std::cout << "67" << std::endl;

    for (size_t i = 0; i < piVector->size(); ++i) {
        mPiVector->push_back((*piVector)[i]);
    }
    	// std::cout << "72" << std::endl;

    for (size_t i = 0; i < tauVector->size(); ++i) {
        mTauVector->push_back((*tauVector)[i]);
    }
    	// std::cout << "77" << std::endl;

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
    	// std::cout << "96" << std::endl;

    this->computeNonSingeltons(pSubset);
    	// std::cout << "99" << std::endl;

    vsize_t*  neighbors = mBloomierHash->getKNeighbors(pKey, mK, mM);
    	// std::cout << "102" << std::endl;

    for (auto it = neighbors->begin(); it != neighbors->end(); ++it) {
        if (mNonSingeltons->find((*it)) == mNonSingeltons->end()) {
            return i;
        }
        ++i;
    } 
    return -1;
}

void OrderAndMatchFinder::computeNonSingeltons(vsize_t* pKeyValues) {
    	// std::cout << "114" << std::endl;

    for (auto it = pKeyValues->begin(); it != pKeyValues->end(); ++it) {
        	// std::cout << "117" << std::endl;

        vsize_t* neighbors = mBloomierHash->getKNeighbors((*it), mK, mM);
        	// std::cout << "120" << std::endl;

        for (auto itNeighbors = neighbors->begin(); itNeighbors != neighbors->end(); ++itNeighbors){
            	// std::cout << "123" << std::endl;

            if (mHashesSeen->find((*itNeighbors)) != mHashesSeen->end()) {
                	// std::cout << "125" << std::endl;

                mNonSingeltons->insert((*itNeighbors));
            }
        }
        	// std::cout << "131" << std::endl;

        for (auto itNeighbors = neighbors->begin(); itNeighbors != neighbors->end(); ++itNeighbors){
            mHashesSeen->insert((*itNeighbors));
        }
    }
}