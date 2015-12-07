#include "../typeDefinitions.h"
#include <set>
#include <gtest/gtest_prod.h>
#include "bloomierHash.h"

#ifndef ORDER_AND_MATCH_FINDER_H
#define ORDER_AND_MATCH_FINDER_H
class OrderAndMatchFinder {
  private:
    size_t mModulo;
    size_t mNumberOfElements;
    size_t mBloomFilterInstance;
    size_t mBloomFilterInstanceDifferentSeed;
    size_t mBloomFilterHashesSeen;
    size_t mBloomFilterNonSingeltons;
    size_t mBloomFilterSeed;
    vsize_t* mPiVector;
    vsize_t* mTauVector;
    
    BloomierHash* mBloomierHash;
    std::unordered_map<size_t, size_t>* mSeeds;
    Hash* mHash;
    bool findMatch(vsize_t* pSubset);
    FRIEND_TEST(OrderAndMatchFinderTest, computeNonSingeltonsTest);
    void computeNonSingeltons(vsize_t* pKeyValues, size_t pSeed = 0);
    FRIEND_TEST(OrderAndMatchFinderTest, tweakTest);
    int tweak(size_t pKey, vsize_t* pSubset, size_t pSeed = 0);
  public:
  	OrderAndMatchFinder(size_t pModulo, size_t pNumberOfElements, BloomierHash* pBloomierHash);
    ~OrderAndMatchFinder();
    void find(vsize_t* pSubset);
    vsize_t* getPiVector();
    vsize_t* getTauVector();
    size_t getSeed(size_t pKey);
};
#endif // ORDER_AND_MATCH_FINDER_H