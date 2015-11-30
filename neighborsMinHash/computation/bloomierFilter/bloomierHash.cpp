#include "bloomierHash.h"

BloomierHash::BloomierHash(size_t pM, size_t pK, size_t pBitVectorSize) {
	mM = pM;
	mK = pK;
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
vsize_t* BloomierHash::getKNeighbors(size_t pElement, size_t pK, size_t pModulo) {
	// std::cout << "22" << std::endl;
	vsize_t* kNeighbors = new vsize_t(pK);
	for (size_t i = 0; i < pK; ++i) {
			// std::cout << "i: "<< i << std::endl;

		(*kNeighbors)[i] = mHash->hash(pElement+1, i+1, pModulo);
	}
	return kNeighbors;
};