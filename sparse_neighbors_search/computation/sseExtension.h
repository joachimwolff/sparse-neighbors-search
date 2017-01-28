// #include <xmmintrin.h>
#include <emmintrin.h>
#ifndef SSE_EXTENSION
#define SSE_EXTENSION
// source: http://stackoverflow.com/questions/10500766/sse-multiplication-of-4-32-bit-integers
// if sse4.1 support is given function is build in
static inline __m128i _mm_mul_epi32_4int(const __m128i &a, const __m128i &b) {
    __m128i tmp1 = _mm_mul_epu32(a,b); /* mul 2,0*/
    __m128i tmp2 = _mm_mul_epu32( _mm_srli_si128(a,4), _mm_srli_si128(b,4)); /* mul 3,1 */
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE (0,0,2,0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE (0,0,2,0))); /* shuffle results to [63..0] and pack */
}
static inline __m128i _mm_not_si128 (const __m128i &x) {
    // Returns ~x, the bitwise complement of x:
    return _mm_xor_si128(x, _mm_cmpeq_epi32(_mm_setzero_si128(), _mm_setzero_si128()));
}

// returns the changed armins if there is a new minHash value at position 0, .., 3
static inline __m128i _mm_argmin_change_epi32 (const __m128i &pArgmin, const __m128i &pMinimumVector, const __m128i &pHashValue, const __m128i &pArgminValue) {
    __m128i compareResult = _mm_cmpeq_epi32(pMinimumVector, pHashValue);
    return _mm_or_si128(_mm_and_si128(compareResult, pArgminValue), _mm_andnot_si128(compareResult, pArgmin));
}


// inspired by https://stackoverflow.com/questions/9877700/getting-max-value-in-a-m128i-vector-with-sse/9878321#9878321
static inline uint32_t _mm_get_argmin(const __m128i &pArgmin, const __m128i &pMinHashValues) {

    __m128i max1 = _mm_shuffle_epi32(pMinHashValues, _MM_SHUFFLE(0,0,3,2));
    __m128i max2 = _mm_min_epi32(pMinHashValues,max1);
    __m128i max3 = _mm_shuffle_epi32(max2, _MM_SHUFFLE(0,0,0,1));
    __m128i max4 = _mm_min_epi32(max2,max3);
    int minValue = _mm_cvtsi128_si32(max4);
    // std::cout << minValue << std::endl;
     __m128i compare = _mm_setr_epi32(minValue, minValue, minValue, minValue);
     __m128i argmin = _mm_and_si128(pArgmin, _mm_cmpeq_epi32(pMinHashValues, compare));
    max1 = _mm_shuffle_epi32(argmin, _MM_SHUFFLE(0,0,3,2));
    max2 = _mm_max_epi32(argmin,max1);
    max3 = _mm_shuffle_epi32(max2, _MM_SHUFFLE(0,0,0,1));
    max4 = _mm_max_epi32(max2,max3); 
    return (uint32_t) _mm_cvtsi128_si32(max4);
}
#endif // SSE_EXTENSION 