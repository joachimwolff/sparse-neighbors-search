#include "bloomierHash.h"
#include <set>
#include <bitset>
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
    char foo;
	srand(mHashSeed*pKey);
	size_t randValue = rand();// % (255*mBitVectorSize);
	// std::cout << "randValue: " << randValue << std::endl;
	for (size_t i = 0; i < mBitVectorSize; ++i) {
		(*mask)[i] =  static_cast<unsigned char>((randValue >> (8*i))& 0b00000000000000000000000011111111);
        // (*mask)[i] = (*m
        // foo = static_cast<char>( randValue >> (8*i));
        std::bitset<8> peter((*mask)[i]);
        // std::cout << "peter: " << peter << std::endl;
		// std::cout << "randValue2: " << std::bitset<bits>(*mask)[i]) << std::endl;
		// std::cout << "randValueFoo: " << static_cast<size_t>((*mask)[i]) << std::endl;
        // std::cout << "sizeofMASK: " << sizeof((*mask)[i]) << std::endl;
        
        // std::cout << "sizeofFOO: " << sizeof(foo) << std::endl;
        
	}
	
	return mask;
};
vsize_t* BloomierHash::getKNeighbors(size_t pElement) {
	vsize_t* kNeighbors = new vsize_t(mNumberOfElements);
	std::set<size_t>* setOfNeighbors = new std::set<size_t>();
	// srand(pElement);
    size_t seedChange = 1;
	for (size_t i = 0; i < mNumberOfElements; ++i) {
		size_t neighbor = mHash->hash(pElement+1, mHashSeed*mHashSeed, mModulo);
        size_t size = setOfNeighbors->size();
        setOfNeighbors->insert(neighbor);
        while (size == setOfNeighbors->size()) {
            // std::cout << "neighbor: " << neighbor << std::endl;
            neighbor = mHash->hash(pElement+1, (mHashSeed+seedChange)*(mHashSeed+seedChange), mModulo);
            setOfNeighbors->insert(neighbor);
            // std::cout << "neighbor2: " << neighbor << std::endl;
            
            ++seedChange;
        }
        seedChange = 1;
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