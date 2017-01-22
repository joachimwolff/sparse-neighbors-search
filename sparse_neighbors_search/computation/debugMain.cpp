#include "hash.h"
#include <vector>
// #include <cout>
#include <iostream>
#include <emmintrin.h>

int main(int** args) {
    Hash* hash = new Hash();
    int a[4] = {1, 67, 800, 1212123};

    std::vector<uint32_t> foo = {1, 67, 800, 1212123};
    // __m128i foo_SSE = _mm_setzero_si128();
    __m128i foo_SSE = _mm_load_epi32(a);
     
    __m128i seed_SSE = _mm_set_epi32(1, 1, 1, 1);
    __m128i foo_result_SSE = hash->hash_SSE(foo_SSE, seed_SSE);
    
    for(size_t i = 0; i < 4; ++i) {
        std::cout << "\n\nNumber " << i << ": \t" << foo[i] << std::endl;
        std::cout << "NumberSSE " << i << ": \t" << foo_SSE[i] << std::endl;
        
        // std::cout << "\n\nTraditional: " << hash->hash(foo[i], 1, MAX_VALUE) << std::endl;
        // std::cout << "SSE: " << foo_result_SSE[i] % MAX_VALUE << std::endl;
    }
  
    delete hash;
}