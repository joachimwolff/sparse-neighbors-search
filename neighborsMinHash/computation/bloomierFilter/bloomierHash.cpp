#include "bloomierHash.h"
#include <set>
BloomierHash::BloomierHash(size_t pModulo,size_t pNumberOfElements, size_t pBitVectorSize) {
	mModulo = pModulo;
	mNumberOfElements = pNumberOfElements;
	// mQ = pQ;
	mHash = new Hash();
	mBitVectorSize = pBitVectorSize;
};
BloomierHash::~BloomierHash() {

};
bitVector* BloomierHash::getMask(size_t pKey) {
	bitVector* mask = new bitVector(mBitVectorSize);
	srand(pKey);
	size_t randValue = rand();
	for (size_t i = 0; i < mBitVectorSize; ++i) {
		(*mask)[i] = randValue >> (i*8);
	}
	return mask;
};
vsize_t* BloomierHash::getKNeighbors(size_t pElement) {
	vsize_t* kNeighbors = new vsize_t(mNumberOfElements);
	// std::set<size_t>* setOfNeighbors = new std::set<size_t>();
	// srand(pElement);
	for (size_t i = 0; i < mNumberOfElements; ++i) {
		size_t neighbor = mHash->hash(pElement+1, rand(), mModulo);
		(*kNeighbors)[i] = neighbor;
	}
	// delete setOfNeighbors;
	return kNeighbors;
};