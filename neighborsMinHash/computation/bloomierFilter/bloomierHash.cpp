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
void BloomierHash::getKNeighbors(size_t pElement, size_t pSeed, vsize_t* pNeighbors) {
    size_t seedValue = pSeed;
    if (seedValue == 0) {
        seedValue = mHashSeed;
    }
    size_t seedChange = 1;
    bitVector* bloomFilterValueSeen = new bitVector(ceil(mModulo/8.0));
	for (size_t i = 0; i < pNeighbors->size(); ++i) {
		size_t neighbor = mHash->hash(pElement+1, seedValue+seedChange, mModulo);
        unsigned char index = floor(neighbor / 8.0);
        unsigned char value = 1 << (neighbor % 8);
        unsigned char valueSeen = (*bloomFilterValueSeen)[index] & value;
        while (value == valueSeen) {
            ++seedChange;
            neighbor = mHash->hash(pElement+1, (seedValue+seedChange)*(seedValue+seedChange), mModulo);
            index = floor(neighbor / 8.0);
            value = 1 << (neighbor % 8);
            valueSeen = (*bloomFilterValueSeen)[index] & value;
        }
        (*bloomFilterValueSeen)[index] = (*bloomFilterValueSeen)[index] | value;
        
        seedChange = 1;
		(*pNeighbors)[i] = neighbor;
		++pElement;
		
	}
    delete bloomFilterValueSeen;
};

size_t BloomierHash::getHashSeed() {
	return mHashSeed;
};

void BloomierHash::setHashSeed(size_t pHashSeed) {
	mHashSeed = pHashSeed;
}