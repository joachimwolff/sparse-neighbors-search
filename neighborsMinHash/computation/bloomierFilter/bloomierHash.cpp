#include "bloomierHash.h"
#include <set>
#include <bitset>
BloomierHash::BloomierHash(size_t pModulo,size_t pNumberOfElements, size_t pBitVectorSize, size_t pHashSeed) {
	mModulo = pModulo;
	mNumberOfElements = pNumberOfElements;
	// mQ = pQ;
	mHash = new Hash();
	mBitVectorSize = pBitVectorSize;
	mHashSeed = pHashSeed;
    
};
BloomierHash::~BloomierHash() {

};
bitVector* BloomierHash::getMask(size_t pKey) {
	bitVector* mask = new bitVector(mBitVectorSize);
	srand(mHashSeed*pKey);
	size_t randValue = rand();
	for (size_t i = 0; i < mBitVectorSize; ++i) {
		(*mask)[i] =  static_cast<unsigned char>((randValue >> (8*i))& 0b00000000000000000000000011111111);
	}
	
	return mask;
};
vsize_t* BloomierHash::getKNeighbors(size_t pElement, size_t pSeed) {
    size_t seedValue = pSeed;
    if (seedValue == 0) {
        seedValue = mHashSeed;
    }
	vsize_t* kNeighbors = new vsize_t(mNumberOfElements);
	std::set<size_t>* setOfNeighbors = new std::set<size_t>();
    size_t seedChange = 1;
	for (size_t i = 0; i < mNumberOfElements; ++i) {
		size_t neighbor = mHash->hash(pElement+1, seedValue+seedChange, mModulo);
        size_t size = setOfNeighbors->size();
        setOfNeighbors->insert(neighbor);
        while (size == setOfNeighbors->size()) {
            neighbor = mHash->hash(pElement+1, (seedValue+seedChange)*(seedValue+seedChange), mModulo);
            setOfNeighbors->insert(neighbor);
            ++seedChange;
        }
        seedChange = 1;
		(*kNeighbors)[i] = neighbor;
		++pElement;
		
	}
	return kNeighbors;
};

size_t BloomierHash::getHashSeed() {
	return mHashSeed;
};

void BloomierHash::setHashSeed(size_t pHashSeed) {
	mHashSeed = pHashSeed;
}