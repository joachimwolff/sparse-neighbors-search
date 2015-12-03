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
    BloomierHash(size_t pModulo, size_t pNumberOfElements, size_t pBitVectorSize, size_t pHashSeed);
    ~BloomierHash();
    bitVector* getMask(size_t pKey);
    vsize_t* getKNeighbors(size_t pElement);
    size_t getHashSeed();
    void setHashSeed(size_t pHashSeed);
};
#endif // BLOOMIER_HASH_H