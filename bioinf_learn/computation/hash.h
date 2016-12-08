#include "typeDefinitions.h"
#include <functional>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <intrin.h>
#ifndef HASH_H
#define HASH_H
class Hash {
  private:
    const double A = sqrt(2) - 1;
    _mm_not_si128 (__m128i x) {
	    // Returns ~x, the bitwise complement of x:
	    return _mm_xor_si128(x, _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()));
    }
    __m128i hash_SSE(__m128i keys, _m128i aModulo) {
        // key = key * A;
        __m128i keys_temp = {A, A, A, A};
        keys = _mm_mul_epi32(keys_temp, keys);
        
        // key = ~key + (key << 15); 
        keys_temp = keys;
        // negate keys_temp
        keys_temp = _mm_not_si128(keys_temp);
        // left shift 15 bits
        keys = _mm_slli_epi32(keys, 15);
        keys = _mm_add_epi32(keys_temp, keys);

        // key = key ^ (key >> 12);
        keys_temp = keys;
        keys = _mm_srli_epi32(keys, 12);
        keys = _mm_xor_si128(keys_tmp, keys);

        // key = key + (key << 2);
        keys_temp = keys;
        keys = _mm_slli_epi32(keys, 2);
        keys = _mm_add_epi32(keys_temp, keys);

        // key = key ^ (key >> 4);
        keys_temp = keys;
        keys = _mm_srli_epi32(keys, 4);
        keys = _mm_xor_si128(keys_tmp, keys);           

        // key = key * 2057;
        keys_temp = {2057, 2057, 2057, 2057};
        keys = _mm_mul_epi32(keys_temp, keys);

        // key = key ^ (key >> 16);
        keys_temp = keys;
        keys = _mm_srli_epi32(keys, 16);
        keys = _mm_xor_si128(keys_tmp, keys); 
        // return key % aModulo;
        
        

        return keys;

    }
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