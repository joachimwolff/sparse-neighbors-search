#include <unordered_map>
#include "orderAndMatchFinder.h"

OrderAndMatchFinder::OrderAndMatchFinder(size_t pM, size_t pK, size_t pQ) {
    mM = pM;
    mK = pK;
    mQ = pQ;
    piVector = new vsize_t();
    tauVector = new vsize_t();
}
OrderAndMatchFinder::~OrderAndMatchFinder() {
    
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
    
    return 0;
}