#include "bloomierFilter.h"

BloomierFilter::BloomierFilter(size_t pModulo, size_t pNumberOfElements, size_t pBitWidth, size_t pHashSeed, size_t pMaxBinSize){
	mModulo = pModulo;
	mNumberOfElements = pNumberOfElements;
	// mQ = pBitWidth;
    mHashSeed = pHashSeed;
	mBitVectorSize = ceil(pBitWidth / static_cast<float>(CHAR_BIT));
	mBloomierHash = new BloomierHash(pModulo, pNumberOfElements, mBitVectorSize, pHashSeed);
	mOrderAndMatchFinder = new OrderAndMatchFinder(pModulo, pNumberOfElements, mBloomierHash);
	mTable = new bloomierTable(pModulo);
	mValueTable = new vvsize_t_p(pModulo);
	for (size_t i = 0; i < pModulo; ++i) {
		(*mTable)[i] = new bitVector(mBitVectorSize, 0);
        (*mValueTable)[i] = new vsize_t();
	}
    mEncoder = new Encoder(mBitVectorSize);
	mPiIndex = 0;
    mMaxBinSize = pMaxBinSize;
}

BloomierFilter::~BloomierFilter(){
    delete mEncoder;
    for (size_t i = 0; i < mTable->size(); ++i) {
		delete (*mTable)[i];
	}
    delete mTable;
    delete mValueTable;
    delete mOrderAndMatchFinder;
    delete mBloomierHash;
}
void BloomierFilter::xorOperation(bitVector* pValue, bitVector* pMask, vsize_t* pNeighbors) {
    this->xorBitVector(pValue, pMask);
	for (auto it = pNeighbors->begin(); it != pNeighbors->end(); ++it) {
		if (*it < mTable->size()) {
			this->xorBitVector(pValue, (*mTable)[(*it)]);
		}
	}
}
vsize_t* BloomierFilter::get(size_t pKey) {
    
    size_t valueSeed = mOrderAndMatchFinder->getSeed(pKey);
    if (valueSeed == MAX_VALUE) {
        return new vsize_t();
    } else if (valueSeed == MAX_VALUE - 1) {
        valueSeed = mHashSeed;
    }
    vsize_t* neighbors = new vsize_t(mNumberOfElements);
	mBloomierHash->getKNeighbors(pKey, valueSeed, neighbors);
	bitVector* mask = mBloomierHash->getMask(pKey);
		
	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);

	size_t h = mEncoder->decode(valueToGet);
	if (h < neighbors->size()) {
		size_t L = (*neighbors)[h];
		if (L < mValueTable->size()) {
			
            delete neighbors;
			delete mask;
			delete valueToGet;
			return (*mValueTable)[L];
		}
	}
	
    delete neighbors;
	delete mask;
	delete valueToGet;
	return new vsize_t();
}
bool BloomierFilter::set(size_t pKey, size_t pValue) {
    size_t valueSeed = mOrderAndMatchFinder->getSeed(pKey);
    if (valueSeed == MAX_VALUE) {
        // new value
		this->create(pKey, pValue);
		return true;
    } else if (valueSeed == MAX_VALUE - 1) {
        // value was before there, used default hash seed
        valueSeed = mHashSeed;
    }
    // else: a different hash seed was used
    vsize_t* neighbors = new vsize_t(mNumberOfElements);
    
    mBloomierHash->getKNeighbors(pKey, valueSeed, neighbors);
	bitVector* mask = mBloomierHash->getMask(pKey);
	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);
	size_t h = mEncoder->decode(valueToGet);
	
	delete mask;
	delete valueToGet;
	if (h < neighbors->size()) {
		size_t L = (*neighbors)[h];
		delete neighbors;
		if (L < mValueTable->size()) {
			vsize_t* v = (*mValueTable)[L];
            if (v->size() < mMaxBinSize) {
                if (v->size() > 0) {
                    v->push_back(pValue);
                }
            } else {
                v->clear();
            }
			return true;
		}
	}
    delete neighbors;
	return false;
}

void BloomierFilter::create(size_t pKey, size_t pValue) {
	
    vsize_t* neighbors = mOrderAndMatchFinder->findIndexAndReturnNeighborhood(pKey);
    vsize_t* piVector = mOrderAndMatchFinder->getPiVector();
	vsize_t* tauVector = mOrderAndMatchFinder->getTauVector();

    size_t key = (*piVector)[mPiIndex];
    bitVector* mask = mBloomierHash->getMask(key);
    
    size_t l = (*tauVector)[mPiIndex];
    size_t L = (*neighbors)[l];
    bitVector* encodeValue = mEncoder->encode(l);
    this->xorBitVector((*mTable)[L], encodeValue);
    this->xorBitVector((*mTable)[L], mask);
    for (size_t j = 0; j < neighbors->size(); ++j) {
        if (j != l) {
            this->xorBitVector((*mTable)[L], (*mTable)[(*neighbors)[j]]);
        }
    }
    vsize_t* valueVector = new vsize_t(1);
    (*valueVector)[0] = pValue;
    (*mValueTable)[L] = valueVector;
    delete neighbors;
    delete mask;
    delete encodeValue;
	++mPiIndex;
}

void BloomierFilter::xorBitVector(bitVector* pResult, bitVector* pInput) {
	size_t length = std::min(pResult->size(), pInput->size());
	for (size_t i = 0; i < length; ++i) {
		(*pResult)[i] = (*pResult)[i] ^ (*pInput)[i];
	}
}