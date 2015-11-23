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
    BloomierHash(size_t pM, size_t pK, size_t pBitVectorSize) {
        mM = pM;
        mP = pK;
        mQ = pQ;
        mHash = new Hash();
        mBitVectorSize = pBitVectorSize;
    };
    ~BloomierHash() {

    };
    bitVector* getMask(size_t pKey) {
        bitVector* mask = new bitVector(mBitVectorSize);
        for (size_t i = 0; i < mBitVectorSize; ++i) {
            mask[i] =  static_cast<char>(rand(pKey) % 255);
        }
        return mask;
    };
    vsize_t* getKNeighbors(size_t pElement, size_t pK, size_t pModulo) {
        
        vsize_t* kNeighbors = new vsize_t(pK);
        for (size_t i = 0; i < pK; ++i) {
            (*kNeighbors)[i] = mHash->hash(pElement+1, pModulo, i+1);
        }
        return kNeighbors;
    };

};
#endif // BLOOMIER_HASH_H