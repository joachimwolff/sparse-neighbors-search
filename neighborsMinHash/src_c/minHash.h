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
typedef std::vector<size_t> vsize_t;
// typedef std::map< size_t, vsize_t > mapVector;
typedef std::unordered_map< size_t, vsize_t > umapVector;
typedef std::vector<vsize_t > vvsize_t;
typedef std::vector< std::vector<float> > vvfloat;
typedef std::vector<std::map<size_t, size_t> > vmSize_tSize_t;
typedef std::unordered_map<size_t, std::pair<size_t, vsize_t> > umap_pair_vector;

#define MAX_VALUE 2147483647

class MinHash {
  private:
    //const size_t MAX_VALUE = 2147483647; 
	  const double A = sqrt(2) - 1;

    size_t numberOfHashFunctions;
    size_t blockSize;
    size_t numberOfCores;
    size_t chunkSize;
    size_t maxBinSize;
    size_t sizeOfNeighborhood;
    size_t minimalBlocksInCommon;
    size_t excessFactor;
    size_t maximalNumberOfHashCollisions;
    umap_pair_vector* signatureStorage;
  	std::vector<umapVector >* inverseIndex;
  	
  	// Return an hash value for a given key in defined range aModulo
  	size_t _size_tHashSimple(size_t key, size_t aModulo) {
  	    key = ~key + (key << 15);
  	    key = key ^ (key >> 12);
  	    key = key + (key << 2);
  	    key = key ^ (key >> 4);
  	    key = key * 2057;
  	    key = key ^ (key >> 16);
  	    return key % aModulo;
  	};
  public:
  	MinHash(size_t pNumberOfHashFunctions, size_t pBlockSize,
                    size_t pNumberOfCores, size_t pChunkSize,
                    size_t pMaxBinSize,
                    size_t pSizeOfNeighborhood, size_t pMinimalBlocksInCommon,
                    size_t pExcessFactor, size_t pMaximalNumberOfHashCollisions);
  	~MinHash();
    // compute the complete inverse index for all given instances and theire non null features
    void computeInverseIndex(const umapVector& instanceFeatureVector);
	  // computes the signature of a given instance
    vsize_t computeSignature(const vsize_t& instanceFeatureVector);
    umap_pair_vector* computeSignatureMap(const umapVector& instanceFeatureVector);
    // computes the neighborhood for the given instances
    std::pair<vvsize_t , vvfloat > computeNeighbors(umap_pair_vector* signatures);

    umap_pair_vector* getSignatureStorage(){return signatureStorage;}

};