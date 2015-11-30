cmake_minimum_required(VERSION 2.6)
 SET(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS} ../neighborsMinHash/computation ../neighborsMinHash/computation/bloomierFilter)
 
# Link runTests with what we want to test and the GTest and pthread library
add_executable(runTests bloomierFilterTest.cpp ../neighborsMinHash/computation/bloomierFilter/bloomierFilter.cpp ../neighborsMinHash/computation/bloomierFilter/bloomierHash.cpp ../neighborsMinHash/computation/bloomierFilter/encoder.cpp ../neighborsMinHash/computation/bloomierFilter/orderAndMatchFinder.cpp)
target_link_libraries(runTests ${GTEST_LIBRARIES} pthread)