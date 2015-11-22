#include "../typeDefinitions.h"
#ifndef ORDER_AND_MATCH_FINDER_H
#define ORDER_AND_MATCH_FINDER_H
class OrderAndMatchFinder {
  private:
    size_t mM;
    size_t mK;
    size_t mQ
    vsize_t* mPiVector;
    vsize_t* mTauVector;
    bool findMatch(vsize_t* pSubset);
  public:
  	OrderAndMatchFinder(size_t pM, size_t pK, size_t pQ);
    void find(vsize_t* pSubset);
    vsize_t* getPiVector();
    vsize_t* getTauVector();
    size_t tweak (size_t pKey, vsize_t pSubset);
};
#endif // ORDER_AND_MATCH_FINDER_H