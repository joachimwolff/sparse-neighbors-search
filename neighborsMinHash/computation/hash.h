#include "typeDefinitions.h"
#include <functional>
#ifndef HASH_H
#define HASH_H
class Hash {
  private:
    const double A = sqrt(2) - 1;
    
    size_t size_tHashSimple(size_t key, size_t aModulo) {
          // std::hash<size_t> hash;
          // return hash(key);
          key = key * A;
          key = ~key + (key << 15);
          key = key ^ (key >> 12);
          key = key + (key << 2);
          key = key ^ (key >> 4);
          key = key * 2057;
          key = key ^ (key >> 16);
          return key % aModulo;
    }; 
  public:      
    size_t hash(size_t pKey, size_t pSeed, size_t pModulo) {
        return size_tHashSimple(pKey * pSeed, pModulo);
    };
    
    size_t hash_cpp_lib(size_t pKey, size_t pSeed, size_t pModulo) {
        std::hash<size_t> hash_function;
        
        return hash_function(pKey*pSeed) % pModulo;
    }
};
#endif // HASH_H