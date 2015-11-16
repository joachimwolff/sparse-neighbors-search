#include <functional>

#include "../typeDefinitions.h"
#ifndef BLOOMIER_HASH_H
#define BLOOMIER_HASH_H
class BloomierHash {
  private:
    size_t mHashSeed;
    size_t mM;
    size_t mK;
    size_t mQ;
    size_t byteSize;

  public:      
    BloomierHash(size_t pHashSeed, size_t pM, size_t pK, size_t pQ) {

    };
    BloomierHash();
    ~BloomierHash() {

    };
    size_t getHashValue(size_t pKey);
    size_t getM(size_t pKey);


};
#endif // BLOOMIER_HASH_H