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
#include <vector>
#include <map> 
#include <unordered_map>

#include "minHash.h"

#define MAX_VALUE 2147483647

class MinHashBloomierFilter : MinHashBase {
  private:
    //const size_t MAX_VALUE = 2147483647; 
	  const double A = sqrt(2) - 1;

    umap_pair_vector* mSignatureStorage;
  	std::vector<umapVector >* mInverseIndex;

  public:
  	MinHashBloomierFilter(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions);
  	~MinHashBloomierFilter();
    // compute the complete inverse index for all given instances and theire non null features
    void computeInverseIndex(const umapVector& instanceFeatureVector);
	  // computes the signature of a given instance
    vsize_t computeSignature(const vsize_t& instanceFeatureVector);
    umap_pair_vector* computeSignatureMap(const umapVector& instanceFeatureVector);
    // computes the neighborhood for the given instances
    std::pair<vvsize_t , vvfloat > computeNeighbors(const umap_pair_vector* signatures, const size_t doubleElements);

    umap_pair_vector* getSignatureStorage(){return signatureStorage;}
    size_t getDoubleElementsStorage(){return mDoubleElementsStorage;}
    size_t getDoubleElementsQuery(){return mDoubleElementsQuery;}

};