/**
 Copyright 2015 Joachim Wolff
 Master Project
 Tutors: Milad Miladi, Fabrizio Costa
 Summer semester 2015

 Chair of Bioinformatics
 Department of Computer Science
 Faculty of Engineering
 Albert-Ludwig-University Freiburg im Breisgau
**/

#include "minHashBase.h"


class MinHash : MinHashBase {
  private:


  public:
  	MinHash(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions);
  	~MinHash();
    // Calculate the neighbors inside a given radius.
    csr_neighborhood radiusNeighbors();
    // Calculate the neighbors inside a given radius as a graph.
    csr_neighborhood radiusNeighborsGraph();
    // Fits and calculates the neighbors inside a given radius.
    csr_neighborhood fitRadiusNeighbors();
    // Fits and calculates the neighbors inside a given radius as a graph.
    csr_neighborhood fitRadiusNeighborsGraph();
};