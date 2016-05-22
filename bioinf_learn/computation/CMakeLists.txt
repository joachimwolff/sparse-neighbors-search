cmake_minimum_required(VERSION 2.6)
 SET(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
 
#add_definitions(-DOPENMP)

#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()
set(CMAKE_BUILD_TYPE Release)
#set (CMAKE_CXX_FLAGS_DEBUG)

# Locate GTest
#find_package(GTest REQUIRED)
#find_package(PythonLibs REQUIRED)
#include_directories(${PYTHON_INCLUDE_DIRS} bloomierFilter)

# Link runTests with what we want to test and the GTest and pthread library
add_executable(fit_kneighbors fit_kneighbors.cpp hash.h inverseIndex.cpp inverseIndexStorageUnorderedMap.cpp nearestNeighbors.cpp)
target_link_libraries(fit_kneighbors ${PYTHON_LIBRARIES})
