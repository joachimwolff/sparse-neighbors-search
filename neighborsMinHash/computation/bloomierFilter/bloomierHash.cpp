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
	for (size_t i = 0; i < mBitVectorSize; ++i) {
		srand(pKey);
		(*mask)[i] =  static_cast<char>(rand() % 255);
	}
	return mask;
};
vsize_t* BloomierHash::getKNeighbors(size_t pElement, size_t pK, size_t pModulo) {
	
	vsize_t* kNeighbors = new vsize_t(pK);
	for (size_t i = 0; i < pK; ++i) {
		(*kNeighbors)[i] = mHash->hash(pElement+1, pModulo, i+1);
	}
	return kNeighbors;
};