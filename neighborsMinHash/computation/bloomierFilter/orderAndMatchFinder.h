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
    bitVector* mBloomFilterInstance;
    bitVector* mBloomFilterInstanceDifferentSeed;
    bitVector* mBloomFilterHashesSeen;
    bitVector* mBloomFilterNonSingeltons;
    size_t mBloomFilterSeed;
    vsize_t* mPiVector;
    vsize_t* mTauVector;
    
    BloomierHash* mBloomierHash;
    std::unordered_map<size_t, size_t>* mSeeds;
    Hash* mHash;
    void findMatch(size_t pKey, vsize_t* pNeighbors);
    FRIEND_TEST(OrderAndMatchFinderTest, computeNonSingeltonsTest);
    void computeNonSingeltons(vsize_t* pNeighbors);
    FRIEND_TEST(OrderAndMatchFinderTest, tweakTest);
    int tweak(size_t pKey, size_t pSeed, vsize_t* pNeighbors);
  public:
  	OrderAndMatchFinder(size_t pModulo, size_t pNumberOfElements, BloomierHash* pBloomierHash);
    ~OrderAndMatchFinder();
    vsize_t* findIndexAndReturnNeighborhood(size_t key);
    vsize_t* getPiVector();
    vsize_t* getTauVector();
    size_t getSeed(size_t pKey);
};
#endif // ORDER_AND_MATCH_FINDER_H