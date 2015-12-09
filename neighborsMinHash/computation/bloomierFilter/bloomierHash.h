#include <functional>
#include <cmath>

#include <stdlib.h>
#include "../typeDefinitions.h"
#include "../hash.h"

#ifndef BLOOMIER_HASH_H
#define BLOOMIER_HASH_H
class BloomierHash {
  private:
    size_t mModulo;
    size_t mNumberOfElements;
    Hash* mHash;
    size_t mBitVectorSize;
    size_t mHashSeed;

  public:      
    BloomierHash(const size_t pModulo, const size_t pNumberOfElements, const size_t pBitVectorSize, const size_t pHashSeed);
    ~BloomierHash();
    bitVector* getMask(const size_t pKey);
    void getKNeighbors(const size_t pElement, const size_t pSeed, vsize_t* pNeighbors);
    size_t getHashSeed() const;
    void setHashSeed(const size_t pHashSeed);
};
#endif // BLOOMIER_HASH_H