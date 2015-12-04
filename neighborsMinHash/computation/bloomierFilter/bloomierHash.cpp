#include "bloomierHash.h"
#include <set>
BloomierHash::BloomierHash(size_t pModulo,size_t pNumberOfElements, size_t pBitVectorSize, size_t pHashSeed) {
	mModulo = pModulo;
	mNumberOfElements = pNumberOfElements;
	// mQ = pQ;
	mHash = new Hash();
	mBitVectorSize = pBitVectorSize;
	mHashSeed = pHashSeed + 5;
};
BloomierHash::~BloomierHash() {

};
bitVector* BloomierHash::getMask(size_t pKey) {
	bitVector* mask = new bitVector(mBitVectorSize);
	srand(mHashSeed*pKey);
	size_t randValue = rand() % (255*mBitVectorSize);
	std::cout << "randValue: " << randValue << std::endl;
	for (size_t i = 0; i < mBitVectorSize; ++i) {
		(*mask)[i] =  randValue >> (8*;
		std::cout << "randValue2: " << static_cast<size_t>((*mask)[i]) << std::endl;
	}
	
	return mask;
};
vsize_t* BloomierHash::getKNeighbors(size_t pElement) {
	vsize_t* kNeighbors = new vsize_t(mNumberOfElements);
	// std::set<size_t>* setOfNeighbors = new std::set<size_t>();
	// srand(pElement);
	for (size_t i = 0; i < mNumberOfElements; ++i) {
		size_t neighbor = mHash->hash(pElement+1, mHashSeed*mHashSeed, mModulo);
		// std::cout << "pElement+1: " <<  pElement+1 << " mHashSeed: " << mHashSeed*mHashSeed << std::endl;
		// std::cout << "neighbors: " << neighbor << std::endl;
		(*kNeighbors)[i] = neighbor;
		++pElement;
		
	}
	// delete setOfNeighbors;
	return kNeighbors;
};

size_t BloomierHash::getHashSeed() {
	return mHashSeed;
};

void BloomierHash::setHashSeed(size_t pHashSeed) {
	mHashSeed = pHashSeed;
}