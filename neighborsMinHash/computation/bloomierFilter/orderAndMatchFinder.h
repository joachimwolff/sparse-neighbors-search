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
    size_t mSizeOfBloomFilter;
    bitVector* mBloomFilterInstance;
    bitVector* mBloomFilterInstanceDifferentSeed;
    bitVector* mBloomFilterHashesSeen;
    bitVector* mBloomFilterNonSingeltons;
    size_t mBloomFilterSeed;
    vsize_t* mPiVector;
    vsize_t* mTauVector;
    
    BloomierHash* mBloomierHash;
    std::unordered_map<size_t, size_t>* mSeeds;
    const Hash* mHash;
    void findMatch(const size_t pKey, vsize_t* pNeighbors);
    FRIEND_TEST(OrderAndMatchFinderTest, computeNonSingeltonsTest);
    void computeNonSingeltons(const vsize_t* pNeighbors);
    FRIEND_TEST(OrderAndMatchFinderTest, tweakTest);
    int tweak(const size_t pKey, vsize_t* pNeighbors);
  public:
  	OrderAndMatchFinder(const size_t pModulo, const size_t pNumberOfElements, BloomierHash* pBloomierHash);
    ~OrderAndMatchFinder();
    vsize_t* findIndexAndReturnNeighborhood(const size_t key);
    vsize_t* getPiVector() const;
    vsize_t* getTauVector() const;
    bool getValueSeenBefor(const size_t pKey) const;
    void deleteValueInBloomFilterInstance(const size_t pKey);
    size_t getSeed(const size_t pKey) const;
};
#endif // ORDER_AND_MATCH_FINDER_H