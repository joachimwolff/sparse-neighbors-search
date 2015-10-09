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


typedef std::vector<size_t> vsize_t;
typedef std::map< size_t, vsize_t > umapVector;
typedef std::vector<vsize_t > vvsize_t;
typedef std::vector< std::vector<float> > vvfloat;
typedef std::vector<std::map<size_t, size_t> > vmSize_tSize_t;

class sort_map {
  public:
	size_t key;
	size_t val;
};

class MinHash {
  private:
    const size_t MAX_VALUE = 2147483647;
	const double A = sqrt(2) - 1;

	size_t numberOfHashFunctions;
    size_t blockSize;
    size_t numberOfCores;
    size_t chunkSize;
	size_t maxBinSize;
    size_t lazyFitting;
    size_t sizeOfNeighborhood;
    size_t minimalBlocksInCommon;
    size_t excessFactor;
    float maximalNumberOfHashCollisions;
    umapVector signatureStorage;
	std::vector<umapVector > inverseIndex;

	bool mapSortDescByValue(const sort_map& a, const sort_map& b) {
    	return a.val > b.val;
	};
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
  	MinHash(size_t numberOfHashFunctions, size_t blockSize,
                    size_t numberOfCores, size_t chunkSize,
                    size_t maxBinSize, size_t lazyFitting,
                    size_t sizeOfNeighborhood, size_t minimalBlocksInCommon,
                    size_t excessFactor, float maximalNumberOfHashCollisions);
  	~MinHash();
    // compute the complete inverse index for all given instances and theire non null features
	void computeInverseIndex(const umapVector& instanceFeatureVector);
	// computes the signature of a given instance
    vvsize_t computeSignature(const umapVector& instanceFeatureVector);
    // computes the neighborhood for the given instances
    std::pair<vvsize_t , vvfloat > computeNeighbors(const vvsize_t signatures);

};