#include "hash.h"
#include <vector>
// #include <cout>
#include <iostream>
#include <emmintrin.h>

int main(int** args) {
    Hash* hash = new Hash();
    int a[4] = {1, 67, 800, 1212123};

    std::vector<uint32_t> foo = {1, 67, 800, 1337};
    // __m128i foo_SSE = _mm_setzero_si128();
    __m128i foo_SSE = _mm_setr_epi32(1, 67, 800, 1337);
     
    __m128i seed_SSE = _mm_setr_epi32(2, 2, 2, 2);
    __m128i foo_result_SSE = hash->hash_SSE(foo_SSE, seed_SSE);
    // std::cout << "Size: " << foo_SSE->size() << std::endl;
    for(size_t i = 0; i < 4; ++i) {
        // std::cout << "\n\nNumber " << i << ": \t" << foo[i] << std::endl;
        
        std::cout << "Traditional: " << hash->hash(foo[i], 2, MAX_VALUE) << std::endl;
        // std::cout << "SSE: " << foo_result_SSE[i] % MAX_VALUE << std::endl;
    }
        std::cout << "\n\nNumberSSE " << 0 << ": \t" << (uint32_t) _mm_extract_epi32(foo_result_SSE, 0) << std::endl;
        std::cout << "NumberSSE " << 1 << ": \t" << (uint32_t) _mm_extract_epi32(foo_result_SSE, 1) << std::endl;
        std::cout << "NumberSSE " << 2 << ": \t" <<(uint32_t) _mm_extract_epi32(foo_result_SSE, 2) << std::endl;
        std::cout << "NumberSSE " << 3 << ": \t" << (uint32_t) _mm_extract_epi32(foo_result_SSE, 3) << std::endl;
  
    delete hash;
}