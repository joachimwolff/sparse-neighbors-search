/**
 Copyright 2016 Joachim Wolff
 Master Thesis
 Tutor: Fabrizio Costa, Milad Miladi
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwigs-University Freiburg im Breisgau
**/

#ifndef K_SIZE_SORTED_MAP_H
#define K_SIZE_SORTED_MAP_H


class KSizeSortedMap {

  private: 
    std::map<size_t, float>* mKSizeSortedMap;
    size_t mK;

  public:
    KSizeSortedMap(size_t pSizeOfMap) {
        mKSizeSortedMap = new std::map<size_t, float>();
        mK = pSizeOfMap;
    };
    ~KSizeSortedMap() {
        delete mKSizeSortedMap;
    };
    void insert(size_t pInstance, float pValue) {
        if (mKSizeSortedMap->size() <= mK) {
            (*mKSizeSortedMap)[pInstance] = pValue;
        } else {
            auto rit = mKSizeSortedMap->crbegin();
            if (rit->first < pInstance) {
                return;
            } else {
                (*mKSizeSortedMap)[pInstance] = pValue;
                rit = mKSizeSortedMap->rbegin();
                mKSizeSortedMap->erase(rit);
            }
        }
        return;
    };
    size_t getKey(size_t pIndexPosition) {
        auto it = mKSizeSortedMap->begin();
        it += pIndexPosition;
        return it->first;
    };
    float getValue(size_t pIndexPosition) {
        auto it = mKSizeSortedMap->begin();
        it += pIndexPosition;
        return it->second;
    };    

};
#endif // K_SIZE_SORTED_MAP_H
