#include "bloomierHash.h"

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
	for (size_t i = 0; i < mNumberOfElements; ++i) {
		(*kNeighbors)[i] = mHash->hash(pElement+1, i+1, mModulo);
	}
	return kNeighbors;
};