#include "../typeDefinitions.h"
#include <set>
#include "bloomierHash.py"

#ifndef ORDER_AND_MATCH_FINDER_H
#define ORDER_AND_MATCH_FINDER_H
class OrderAndMatchFinder {
  private:
    size_t mM;
    size_t mK;
    size_t mQ
    vsize_t* mPiVector;
    vsize_t* mTauVector;
    std::set<size_t> mHashesSeen;
    std::set<size_t> mNonSingeltons;
    BloomierHash* mBloomierHash;
    bool findMatch(vsize_t* pSubset);
    void computeNonSingeltons(std::map<size_t, size_t> pKeyValue);
    int tweak(size_t pKey, vsize_t pSubset);
  public:
  	OrderAndMatchFinder(size_t pM, size_t pK, size_t pQ, BloomierHash* pBloomierHash);
    ~OrderAndMatchFinder();
    void find(vsize_t* pSubset);
    vsize_t* getPiVector();
    vsize_t* getTauVector();
};
#endif // ORDER_AND_MATCH_FINDER_H