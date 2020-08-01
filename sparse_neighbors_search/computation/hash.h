/**
 Copyright 2016, 2017, 2018, 2019, 2020 Joachim Wolff
 PhD Thesis

 Copyright 2015, 2016 Joachim Wolff
 Master Thesis
 Tutor: Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwigs-University Freiburg im Breisgau
**/

#include "typeDefinitions.h"
#include "sseExtension.h"
// #include "avxExtension.h"

#include <smmintrin.h>
#include <functional>
#ifndef HASH_H
#define HASH_H
class Hash {
  private:
    // const double A = sqrt(2) - 1;
    // const float A_float = sqrt(2) - 1;

    
    __m128i hash_SSE_priv(__m128i keys) {

        // __m128 keys_temp = {A_float, A_float, A_float, A_float};
        // // conversion to float
        // __m128 keys_float = _mm_cvtepi32_ps(keys);
        // // multiplication of key * A
        // keys_float = _mm_mul_ps(keys_float, keys_temp);
        // // conversion back to int
        // keys = _mm_cvttps_epi32(keys_float);
        
        // key = ~key + (key << 15); 
        __m128i keys_temp_int = keys;
        // negate keys_temp
        keys_temp_int = _mm_not_si128(keys_temp_int);
        // left shift 15 bits
        keys = _mm_slli_epi32(keys, 15);
        keys = _mm_add_epi32(keys_temp_int, keys);

        // key = key ^ (key >> 12);
        keys_temp_int = keys;
        keys = _mm_srli_epi32(keys, 12);
        keys = _mm_xor_si128(keys_temp_int, keys);
        
        // key = key + (key << 2);
        keys_temp_int = keys;
        keys = _mm_slli_epi32(keys, 2);
        keys = _mm_add_epi32(keys_temp_int, keys);

        // key = key ^ (key >> 4);
        keys_temp_int = keys;
        keys = _mm_srli_epi32(keys, 4);
        keys = _mm_xor_si128(keys_temp_int, keys);           

        // key = key * 2057;
        uint32_t value = 2057;
        __m128i constant_value = _mm_set_epi32(value, value, value, value);
        keys = _mm_mullo_epi32(constant_value, keys);

        // key = key ^ (key >> 16);
        keys_temp_int = keys;
        keys = _mm_srli_epi32(keys, 16);
        keys = _mm_xor_si128(keys_temp_int, keys); 

        return keys;

    }
    // __m256i hash_SSE_priv_64(__m256i keys) {

    //     __m256i keys_temp_int = keys;
    //     // negate keys_temp
    //     keys_temp_int = _mm256_not_si256(keys_temp_int);
    //     // left shift 15 bits
    //     keys = _mm256_slli_epi64(keys, 21);
    //     keys = _mm256_add_epi64(keys_temp_int, keys);

    //     // key = key ^ (key >> 24);
    //     keys_temp_int = keys;
    //     keys = _mm256_srli_epi64(keys, 24);
    //     keys = _mm256_xor_si256(keys_temp_int, keys);
        
    //     // (key + (key << 3)) + (key << 8); // key * 265
    //     uint64_t value = 256;
    //     __m256i constant_value = _mm256_set_epi64x(value, value, value, value);
    //     keys_temp_int = keys;
    //     __m256i keys_temp_int2 = keys;
    //     keys = _mm256_srli_epi64(keys, 8); // (key << 8)
    //     keys_temp_int2 = _mm256_srli_epi64(keys_temp_int2, 3); //(key << 3)
    //     keys_temp_int = _mm256_add_epi64(keys_temp_int, keys_temp_int2); //(key + (key << 3)) 
    //     keys = _mm256_add_epi64(keys_temp_int, keys); // (key + (key << 3)) + (key << 8)
    //     // keys = _mm256_mullo_epi64(constant_value, keys); // function only in avx 512

    //      // key = key ^ (key >> 14);
    //     keys_temp_int = keys;
    //     keys = _mm256_srli_epi64(keys, 14);
    //     keys = _mm256_xor_si256(keys_temp_int, keys);


    //     // (key + (key << 2)) + (key << 4); // key * 21
    //     value = 21;
    //     constant_value = _mm256_set_epi64x(value, value, value, value);
    //     keys_temp_int = keys;
    //     keys_temp_int2 = keys;
    //     keys = _mm256_srli_epi64(keys, 4); // (key << 4)
    //     keys_temp_int2 = _mm256_srli_epi64(keys_temp_int2, 2); //(key << 2)
    //     keys_temp_int = _mm256_add_epi64(keys_temp_int, keys_temp_int2); //(key + (key << 2)) 
    //     keys = _mm256_add_epi64(keys_temp_int, keys); // (key + (key << 2)) + (key << 4)
    //      // keys = _mm256_mullo_epi64(constant_value, keys); // function only in avx 512

    //     // key = key ^ (key >> 28);
    //     keys_temp_int = keys;
    //     keys = _mm256_srli_epi64(keys, 28);
    //     keys = _mm256_xor_si256(keys_temp_int, keys);
    //     // key = key + (key << 31);
    //     keys = _mm256_slli_epi64(keys, 21);
    //     keys = _mm256_add_epi64(keys_temp_int, keys);
    //     return keys;

    // }

    uint32_t size_tHashSimple(uint32_t key, uint32_t aModulo) {
          // source:  Thomas Wang: Integer Hash Functions, 1997 / 2007 
          // https://gist.github.com/badboy/6267743
        //   key = key * A;
          key = ~key + (key << 15);
          key = key ^ (key >> 12);
          key = key + (key << 2);
          key = key ^ (key >> 4);
          key = key * 2057;
          key = key ^ (key >> 16);
          return key;// % aModulo;
    }; 
    // uint32_t size_tHashSimple_64(uint64_t key, uint64_t aModulo) {
    //     // source:  Thomas Wang: Integer Hash Functions, 1997 / 2007 
    //     // https://gist.github.com/badboy/6267743
        
       
    //     key = (~key) + (key << 21); // key = (key << 21) - key - 1;
    //     key = key ^ (key >> 24);
    //     key = (key + (key << 3)) + (key << 8); // key * 265
    //     key = key ^ (key >> 14);
    //     key = (key + (key << 2)) + (key << 4); // key * 21
    //     key = key ^ (key >> 28);
    //     key = key + (key << 31);

    //     return key;// % aModulo;
    // }; 
    
    short unsigned int shortHashSimple(short unsigned int key, short unsigned int aModulo) {
        //   key = key * A;
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
        return size_tHashSimple(pKey * pSeed, pModulo);
    };
    // uint64_t hash_64(uint64_t pKey, uint64_t pSeed, uint64_t pModulo) {
    //     return size_tHashSimple_64(pKey * pSeed, pModulo);
    // };
    short unsigned int hashShort(short unsigned int pKey, short unsigned int pSeed, short unsigned int pModulo) {
        return shortHashSimple(pKey * pSeed, pModulo);
    };
    
    size_t hash_cpp_lib(size_t pKey, size_t pSeed, size_t pModulo) {
        std::hash<size_t> hash_function;
        return hash_function(pKey*pSeed) % pModulo;
    };
    __m128i hash_SSE(__m128i pKeys, __m128i pSeed) {
        pKeys = _mm_mullo_epi32(pKeys, pSeed);
        return hash_SSE_priv(pKeys);
    }

    // __m256i hash_SSE_64(__m256i pKeys, __m256i pSeed) {
    //     // pKeys = _mm256_mullo_epi64(pKeys, pSeed);
    //     // use xor for seed because mullo is only for avx 512
    //     // pKeys = _mm256_xor_si256(pKeys, pSeed);
    //     return hash_SSE_priv_64(pKeys);
    // }
};
#endif // HASH_H