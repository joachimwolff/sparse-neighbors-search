#include "typeDefinitions.h"
#include <functional>
#include <xmmintrin.h>
#include <emmintrin.h>
// #include <intrin.h>
#ifndef HASH_H
#define HASH_H
class Hash {
  private:
    const double A = sqrt(2) - 1;
    const float A_float = sqrt(2) - 1;
    
    __m128i _mm_not_si128 (__m128i x) {
	    // Returns ~x, the bitwise complement of x:
	    return _mm_xor_si128(x, _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()));
    }
    __m128i hash_SSE_priv(__m128i keys) {

        __m128 keys_temp = {A_float, A_float, A_float, A_float};
        // conversion to float
        __m128 keys_float = _mm_cvtepi32_ps(keys);
        // multiplication of key * A
        keys_float =_mm_mul_ps(keys_float, keys_temp);
        // conversion back to int
        keys = _mm_cvtps_epi32(keys_float);

        
        // // key = ~key + (key << 15); 
        // __m128i keys_temp_int = keys;
        // // negate keys_temp
        // keys_temp_int = _mm_not_si128(keys_temp_int);
        // // left shift 15 bits
        // keys = _mm_slli_epi32(keys, 15);
        // keys = _mm_add_epi32(keys_temp_int, keys);

        // // key = key ^ (key >> 12);
        // keys_temp_int = keys;
        // keys = _mm_srli_epi32(keys, 12);
        // keys = _mm_xor_si128(keys_temp_int, keys);

        // // key = key + (key << 2);
        // keys_temp_int = keys;
        // keys = _mm_slli_epi32(keys, 2);
        // keys = _mm_add_epi32(keys_temp_int, keys);

        // // key = key ^ (key >> 4);
        // keys_temp_int = keys;
        // keys = _mm_srli_epi32(keys, 4);
        // keys = _mm_xor_si128(keys_temp_int, keys);           

        // // key = key * 2057;
        // uint32_t value = 2057;
        // __m128i constant_value = _mm_set_epi32(value, value, value, value);
        // keys = _mm_mul_epi32(constant_value, keys);

        // // key = key ^ (key >> 16);
        // keys_temp_int = keys;
        // keys = _mm_srli_epi32(keys, 16);
        // keys = _mm_xor_si128(keys_temp_int, keys); 

        // return key % aModulo;
        // exit(0);
        return keys;

    }
    uint32_t size_tHashSimple(uint32_t key, uint32_t aModulo) {
          // source:  Thomas Wang: Integer Hash Functions, 1997 / 2007 
          // https://gist.github.com/badboy/6267743
          key = key * A;
        //   key = ~key + (key << 15);
        //   key = key ^ (key >> 12);
        //   key = key + (key << 2);
        //   key = key ^ (key >> 4);
        //   key = key * 2057;
        //   key = key ^ (key >> 16);
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
    uint32_t hash(uint32_t pKey, uint32_t pSeed, uint32_t pModulo) {
        return pKey * pSeed;
        // return size_tHashSimple(pKey * pSeed, pModulo);
    };
    short unsigned int hashShort(short unsigned int pKey, short unsigned int pSeed, short unsigned int pModulo) {
        return shortHashSimple(pKey * pSeed, pModulo);
    };
    
    size_t hash_cpp_lib(size_t pKey, size_t pSeed, size_t pModulo) {
        std::hash<size_t> hash_function;
        
        return hash_function(pKey*pSeed) % pModulo;
    };
    __m128i hash_SSE(__m128i pKeys, __m128i pSeed) {
        pKeys = _mm_mul_epi32(pKeys, pSeed);
        return pKeys;
        // return hash_SSE_priv(pKeys);
    }
};
#endif // HASH_H