#include "typeDefinitions.h"
#include <functional>
#ifndef HASH_H
#define HASH_H
class Hash {
  private:
    const double A = sqrt(2) - 1;
    
    size_t size_tHashSimple(size_t key, size_t aModulo) {
          // source:  Thomas Wang: Integer Hash Functions, 1997 / 2007 
          // https://gist.github.com/badboy/6267743
          key = key * A;
          key = ~key + (key << 15);
          key = key ^ (key >> 12);
          key = key + (key << 2);
          key = key ^ (key >> 4);
          key = key * 2057;
          key = key ^ (key >> 16);
          return key % aModulo;
    }; 
    
    short unsigned int shortHashSimple(short unsigned int key, short unsigned int aModulo) {
          key = key * A;
          key = ~key + (key << 7);
          key = key ^ (key >> 6);
          key = key + (key << 1);
          key = key ^ (key >> 2);
          key = key * 1027;
          key = key ^ (key >> 8);
          return key % aModulo;
    };
  public:      
    size_t hash(size_t pKey, size_t pSeed, size_t pModulo) {
        return size_tHashSimple(pKey * pSeed, pModulo);
    };
    short unsigned int hashShort(short unsigned int pKey, short unsigned int pSeed, short unsigned int pModulo) {
        return shortHashSimple(pKey * pSeed, pModulo);
    };
    
    size_t hash_cpp_lib(size_t pKey, size_t pSeed, size_t pModulo) {
        std::hash<size_t> hash_function;
        
        return hash_function(pKey*pSeed) % pModulo;
    }
};
#endif // HASH_H