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


#include <immintrin.h>
#ifndef AVX_EXTENSION
#define AVX_EXTENSION
// source: http://stackoverflow.com/questions/10500766/sse-multiplication-of-4-32-bit-integers
// if sse4.1 support is given function is build in
// static inline __m256i _mm_mul_epi32_4int(const __m256i &a, const __m256i &b) {
//     __m256i tmp1 = _mm256_mul_epu64(a,b); /* mul 2,0*/
//     __m256i tmp2 = _mm256_mul_epu64( _mm256_srli_si256(a,4), _mm256_srli_si256(b,4)); /* mul 3,1 */
//     return _mm_unpacklo_epi32(_mm256_shuffle_epi64(tmp1, _MM_SHUFFLE (0,0,2,0)), _mm256_shuffle_epi64(tmp2, _MM_SHUFFLE (0,0,2,0))); /* shuffle results to [63..0] and pack */
// }
static inline __m256i _mm256_not_si256 (const __m256i &x) {
    // Returns ~x, the bitwise complement of x:
    return _mm256_xor_si256(x, _mm256_cmpeq_epi64(_mm256_setzero_si256(), _mm256_setzero_si256()));
}

// returns the changed armins if there is a new minHash value at position 0, .., 3
static inline __m256i _mm256_argmin_change_epi64 (const __m256i &pArgmin, const __m256i &pMinimumVector, const __m256i &pHashValue, const __m256i &pArgminValue) {
    
    __m256i compareResult = _mm256_cmpeq_epi64(pMinimumVector, pHashValue);
    return _mm256_or_si256(_mm256_and_si256(compareResult, pArgminValue), _mm256_andnot_si256(compareResult, pArgmin));
}


// inspired by https://stackoverflow.com/questions/9877700/getting-max-value-in-a-m128i-vector-with-sse/9878321#9878321
static inline uint64_t _mm256_get_argmin64(const __m256i &pArgmin, const __m256i &pMinHashValues) {
    // std::cout << __LINE__ << std::endl;

    __m256i max1 = _mm256_shuffle_epi64(pMinHashValues, _MM_SHUFFLE(0,0,3,2));
    __m256i max2 = _mm256_min_epi64(pMinHashValues,max1);
    __m256i max3 = _mm256_shuffle_epi64(max2, _MM_SHUFFLE(0,0,0,1));
    __m256i max4 = _mm256_min_epi64(max2,max3);
    int minValue = _mm256_cvtsi128_si64(max4);
    // std::cout << minValue << std::endl;
    // std::cout << __LINE__ << std::endl;

     __m256i compare = _mm_setr_epi32(minValue, minValue, minValue, minValue);
     __m256i argmin = _mm256_and_si256(pArgmin, _mm256_cmpeq_epi64(pMinHashValues, compare));
    // std::cout << __LINE__ << std::endl;

    max1 = _mm256_shuffle_epi64(argmin, _MM_SHUFFLE(0,0,3,2));
    max2 = _mm256_max_epi64(argmin,max1);
    max3 = _mm256_shuffle_epi64(max2, _MM_SHUFFLE(0,0,0,1));
    max4 = _mm256_max_epi64(max2,max3); 
    // std::cout << __LINE__ << std::endl;

    return (uint64_t) _mm256_cvtsi128_si64(max4);
}


// void print128_num(__m256i var)
// {
//     uint16_t *val = (uint16_t*) &var;
//     printf("Numerical: %i %i %i %i %i %i %i %i \n", 
//            val[0], val[1], val[2], val[3], val[4], val[5], 
//            val[6], val[7]);
// }
#endif // AVX_EXTENSION