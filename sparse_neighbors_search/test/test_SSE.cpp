#include <gtest/gtest.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <intrin.h>
#include "../computation/hash.h"

TEST(Hash_test, hash_SSE) {
    __m128i valuesSSE = {1, 10, 37, 3456781239};
    __m128i seedSSE = {1, 1, 1, 1};
    __m128i moduloSSE = {2147483647, 2147483647, 2147483647, 2147483647};
    
    size_t seed = 1;
    size_t modulo = 2147483647;
    size_t value = 1;
    size_t value1 = 10;
    size_t value2 = 37;
    size_t value3 = 3456781239;

    size_t result = hash(value, seed, modulo);
    size_t result1 = hash(value1, seed, modulo);
    size_t result2 = hash(value2, seed, modulo);
    size_t result3 = hash(value3, seed, modulo);

    __m128i resultSSE = hash_SSE(valuesSSE, seedSSE, moduloSSE);

    EXPECT_EQ(result, resultSSE[0]);
    EXPECT_EQ(result1, resultSSE[1]);
    EXPECT_EQ(result2, resultSSE[2]);
    EXPECT_EQ(result3, resultSSE[3]);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}