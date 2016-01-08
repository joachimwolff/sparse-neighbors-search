/**
 Copyright 2015 Joachim Wolff
 Master Thesis
 Tutors: Milad Miladi, Fabrizio Costa
 Winter semester 2015/2016

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/

#include "minHashBase.h"

#ifndef MIN_HASH_H
#define MIN_HASH_H
class MinHash : public MinHashBase {
  private:

  public:
  	MinHash(size_t pNumberOfHashFunctions, size_t pShingleSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions, 
                    int pFast, int pSimilarity, size_t pBloomierFilter,
                    int pPruneInverseIndex, float pPruneInverseIndexAfterInstance,
                    int pRemoveHashFunctionWithLessEntriesAs);
  	~MinHash();
    // Calculate the neighbors inside a given radius.
    neighborhood radiusNeighbors();
    // Calculate the neighbors inside a given radius as a graph.
    neighborhood radiusNeighborsGraph();
    // Fits and calculates the neighbors inside a given radius.
    neighborhood fitRadiusNeighbors();
    // Fits and calculates the neighbors inside a given radius as a graph.
    neighborhood fitRadiusNeighborsGraph();
};
#endif // MIN_HASH_H