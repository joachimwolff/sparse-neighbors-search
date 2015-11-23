#include <functional>
#include <cmath>

#include <stdlib.h>
#include "../typeDefinitions.h"
#include "../hash.h"

#ifndef BLOOMIER_HASH_H
#define BLOOMIER_HASH_H
class BloomierHash {
  private:
    size_t mHashSeed;
    size_t mM;
    size_t mK;
    size_t mQ;
    Hash* mHash;
    size_t mBitVectorSize;

  public:      
    BloomierHash(size_t pM, size_t pK, size_t pBitVectorSize);
    ~BloomierHash();
    bitVector* getMask(size_t pKey);
    vsize_t* getKNeighbors(size_t pElement, size_t pK, size_t pModulo);
};
#endif // BLOOMIER_HASH_H