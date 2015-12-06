#include <gtest/gtest.h>
#include "../neighborsMinHash/computation/bloomierFilter/bloomierFilter.h"
#include "../neighborsMinHash/computation/bloomierFilter/encoder.h"

::testing::AssertionResult IsInRange(int value, int low, int high) {
  if (value < low)
    return ::testing::AssertionFailure()
        << value << " < lower bound " << low;
  else if (value > high)
    return ::testing::AssertionFailure()
        << value << " > upper bound " << high;
  else
    return ::testing::AssertionSuccess()
        << value << " is in range [" 
        << low << ", " << high << "]";
};

TEST(BloomierFilterTest, setAndGetTest) {
    vsize_t* keys = new vsize_t(3);
    (*keys)[0] = 123;
    (*keys)[1] = 231;
    (*keys)[2] = 236;
    BloomierFilter* bloomierFilter = new BloomierFilter(10, 2, 8, keys, 0);
    
    std::unordered_map<size_t, vsize_t >* pKeyValue = new std::unordered_map<size_t, vsize_t >();
    vsize_t first;
    // vvsize_t allInstances;
    // for (size_t i = 0; i < 300; ++i) {
    //     vsize_t p
    //     for (size_t j = 0; j < 400; ++j) {
            
    //     }

    // }
     for (size_t i = 0; i < 1000; ++i) {
        first.push_back(rand());
    }
    vsize_t second;
     for (size_t i = 0; i < 300; ++i) {
        second.push_back(rand());
    }
     vsize_t third;
     for (size_t i = 0; i < 12; ++i) {
        third.push_back(rand());
    }
    // first.push_back(10);
    
    // second.push_back(20);
   
    // third.push_back(30);
    // third.push_back(40);
    
    (*pKeyValue)[123] = first;
    (*pKeyValue)[231] = second;
    (*pKeyValue)[236] = third;
    
    bloomierFilter->create(pKeyValue, 0);
    
    vsize_t* vectorFirst = bloomierFilter->get(123);
    vsize_t* vectorSecond = bloomierFilter->get(231);
    vsize_t* vectorThird = bloomierFilter->get(236);
    EXPECT_EQ(vectorFirst->size(), 1000);
    EXPECT_EQ(vectorSecond->size(), 300);
    EXPECT_EQ(vectorThird->size(), 12);
    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_EQ(vectorFirst->at(i), first[i]);
    }
    for (size_t i = 0; i < 300; ++i) {
        EXPECT_EQ(vectorSecond->at(i), second[i]);
    }
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(vectorThird->at(i), third[i]);
    }
    // EXPECT_EQ(bloomierFilter->get(2)->at(0), 20);
    // EXPECT_EQ(bloomierFilter->get(3)->at(0), 30);
    // EXPECT_EQ(bloomierFilter->get(3)->at(1), 40);
    delete bloomierFilter;
}
TEST(EncoderTest, encodeTest) {
    Encoder* encodeObj = new Encoder(4);
    vsize_t* maskingValues = encodeObj->getMaskingValues();
    EXPECT_EQ(maskingValues->size(), 4);
    EXPECT_EQ(maskingValues->at(0), 255);
    EXPECT_EQ(maskingValues->at(1), 65280);
    EXPECT_EQ(maskingValues->at(2), 16711680);
    EXPECT_EQ(maskingValues->at(3), 4278190080);
    
    delete encodeObj;
    encodeObj = new Encoder(2);
    
    EXPECT_EQ(encodeObj->encode(5)->size(), 2);
    EXPECT_EQ(encodeObj->encode(5)->at(0), 5);
    EXPECT_EQ(encodeObj->encode(5)->at(1), 0);

    delete encodeObj;
    encodeObj = new Encoder(3);
    
    EXPECT_EQ(encodeObj->encode(1024)->size(), 3);
    EXPECT_EQ(encodeObj->encode(1024)->at(0), 0);
    EXPECT_EQ(encodeObj->encode(1024)->at(1), 4);
    EXPECT_EQ(encodeObj->encode(1024)->at(2), 0);
    unsigned char value = 255;
    EXPECT_EQ(encodeObj->encode(255)->size(), 3);
    EXPECT_EQ(encodeObj->encode(255)->at(0), value);
    EXPECT_EQ(encodeObj->encode(255)->at(1), 0);
    EXPECT_EQ(encodeObj->encode(255)->at(2), 0);

    EXPECT_EQ(encodeObj->encode(257)->size(), 3);    
    EXPECT_EQ(encodeObj->encode(257)->at(0), 1);
    EXPECT_EQ(encodeObj->encode(257)->at(1), 1);
    EXPECT_EQ(encodeObj->encode(257)->at(2), 0);
    
    delete encodeObj;
}

TEST(EncoderTest, decodeTest) {
    Encoder* encodeObj = new Encoder(3);
    bitVector* encodedValue = encodeObj->encode(1024);
    EXPECT_EQ(encodedValue->size(), 3);
    EXPECT_EQ(encodedValue->at(0), 0);
    EXPECT_EQ(encodedValue->at(1), 4);
    EXPECT_EQ(encodedValue->at(2), 0);
    
    EXPECT_EQ(encodeObj->decode(encodedValue), 1024);
    
    delete encodeObj;
    delete encodedValue;
}

TEST(BloomierHashTest, getMaskTest) {
    // BloomierHash* bloomierHash = new BloomierHash(19, 3, 2, 0);
    // bitVector* mask = bloomierHash->getMask(1);
    // EXPECT_EQ(mask->at(0), 103);
    // EXPECT_EQ(mask->at(1), 69);
    // delete bloomierHash;
    // delete mask;
}

TEST(BloomierHashTest, getKNeighborsTest) {
    // size_t numberOfElements = 100;
    // BloomierHash* bloomierHash = new BloomierHash(10, numberOfElements, 2, 0);
    // vsize_t* neighbors = bloomierHash->getKNeighbors(1);

    // EXPECT_EQ(neighbors->size(), numberOfElements);
    // for (size_t i = 0; i < numberOfElements; ++i) {
    //     EXPECT_TRUE(IsInRange(neighbors->at(i), 0, 10));
    // }
    // delete bloomierHash;
    // delete neighbors;
}

TEST(OrderAndMatchFinderTest, findTest) {
    // BloomierHash* bloomierHash = new BloomierHash(10, 5, 5, 10);
    // OrderAndMatchFinder* orderAndMatchFinder = new OrderAndMatchFinder(10, 5, bloomierHash);
    // vsize_t* pSubset = new vsize_t();
    // pSubset->push_back(3);
    // pSubset->push_back(4);
    // pSubset->push_back(1);
    // pSubset->push_back(8);
    // orderAndMatchFinder->find(pSubset);
    // vsize_t* piVector = orderAndMatchFinder->getPiVector();
    // vsize_t* tauVector = orderAndMatchFinder->getTauVector();
    
    // for (size_t i = 0; i < piVector->size(); ++i) {
    //     std::cout << "piVector: " << piVector->at(i) << std::endl;
    // }
    // for (size_t i = 0; i < tauVector->size(); ++i) {
    //     std::cout << "tauVector: " << tauVector->at(i) << std::endl;
    // }
}

TEST(OrderAndMatchFinderTest, tweakTest) {
    // BloomierHash* bloomierHash = new BloomierHash(10, 3, 2, 1);
    // OrderAndMatchFinder* orderAndMatchFinder = new OrderAndMatchFinder(10, 3, bloomierHash);
    // vsize_t* subset = new vsize_t(3);
    // (*subset)[0] = 10;
    // (*subset)[1] = 20;
    // (*subset)[2] = 30;    
    // std::cout << "index: " << orderAndMatchFinder->tweak(10, subset) << std::endl;
    // std::cout << "index: " << orderAndMatchFinder->tweak(20, subset) << std::endl;
    // std::cout << "index: " << orderAndMatchFinder->tweak(30, subset) << std::endl;
//    orderAndMatchFinder->tweak(0, subset);   
  //  orderAndMatchFinder->tweak(0, subset);   
     
    // 0:10, 1:20, 2:30
    // 0 --> 8, 9, 7
    // 1 --> 0, 1, 2
    // 2 --> 8, 2, 3
    // EXPECT_TRUE(IsInRange(SomeFunction(), low, high))
}

TEST(OrderAndMatchFinderTest, computeNonSingeltonsTest) {
    // BloomierHash* bloomierHash = new BloomierHash(19, 3, 2, 0);
    // OrderAndMatchFinder* orderAndMatchFinder = new OrderAndMatchFinder(19, 3, bloomierHash);
}
int main (int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return  RUN_ALL_TESTS();
}