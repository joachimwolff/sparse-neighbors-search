#include "bloomierFilter.h"
#ifdef OPENMP
#include <omp.h>
#endif
BloomierFilter::BloomierFilter(const size_t pModulo, const size_t pNumberOfElements, const size_t pBitWidth, const size_t pHashSeed, const size_t pMaxBinSize){
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
void BloomierFilter::xorOperation(bitVector* pValue, const bitVector* pMask, const vsize_t* pNeighbors) {
    this->xorBitVector(pValue, pMask);
	for (auto it = pNeighbors->begin(); it != pNeighbors->end(); ++it) {
		if (*it < mTable->size()) {
			this->xorBitVector(pValue, (*mTable)[(*it)]);
		}
	}
}
vsize_t* BloomierFilter::get(const size_t pKey) {
    
    size_t valueSeed = mOrderAndMatchFinder->getSeed(pKey);
    if (valueSeed == MAX_VALUE) {
        return NULL;
    } else if (valueSeed == MAX_VALUE - 1) {
        valueSeed = mHashSeed;
    }
    vsize_t* neighbors = new vsize_t(mNumberOfElements);
	mBloomierHash->getKNeighbors(pKey, valueSeed, neighbors);
	const bitVector* mask = mBloomierHash->getMask(pKey);
    
	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);

	const size_t h = mEncoder->decode(valueToGet);
	if (h < neighbors->size()) {
		const size_t L = (*neighbors)[h];
		if (L < mValueTable->size()) {
            delete neighbors;
			delete mask;
			delete valueToGet;
            if ((*mValueTable)[L] != NULL) return (*mValueTable)[L];
            return NULL;
		}
	}
    delete neighbors;
	delete mask;
	delete valueToGet;
	return NULL;
}
bool BloomierFilter::set(const size_t pKey, const size_t pValue) {
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
	const bitVector* mask = mBloomierHash->getMask(pKey);
	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);
	const size_t h = mEncoder->decode(valueToGet);
	
	delete mask;
	delete valueToGet;
	if (h < neighbors->size()) {
		const size_t L = (*neighbors)[h];
		delete neighbors;
		if (L < mValueTable->size()) {
#pragma omp critical
            {
                vsize_t* v = (*mValueTable)[L];
                if (v != NULL && v->size() < mMaxBinSize) {
                    if (v->size() > 0) {
                        v->push_back(pValue);
                    }
                } else {
                    // mOrderAndMatchFinder->deleteValueInBloomFilterInstance(pKey);
                    v->clear();
                }
            }
            return true;
            
		}
	}
    delete neighbors;
	return false;
}

void BloomierFilter::create(const size_t pKey, const size_t pValue) {
	
    const vsize_t* neighbors = mOrderAndMatchFinder->findIndexAndReturnNeighborhood(pKey);
    const vsize_t* piVector = mOrderAndMatchFinder->getPiVector();
	const vsize_t* tauVector = mOrderAndMatchFinder->getTauVector();

    const size_t key = (*piVector)[mPiIndex];
    const bitVector* mask = mBloomierHash->getMask(key);
    
    const size_t l = (*tauVector)[mPiIndex];
    const size_t L = (*neighbors)[l];
    const bitVector* encodeValue = mEncoder->encode(l);
    this->xorBitVector((*mTable)[L], encodeValue);
    this->xorBitVector((*mTable)[L], mask);
    for (size_t j = 0; j < neighbors->size(); ++j) {
        if (j != l) {
            this->xorBitVector((*mTable)[L], (*mTable)[(*neighbors)[j]]);
        }
    }
    vsize_t* valueVector = new vsize_t(1);
    (*valueVector)[0] = pValue;
#pragma omp critical
    {
        (*mValueTable)[L] = valueVector;
    }
    delete neighbors;
    delete mask;
    delete encodeValue;
	++mPiIndex;
}

void BloomierFilter::xorBitVector(bitVector* pResult, const bitVector* pInput) {
	const size_t length = std::min(pResult->size(), pInput->size());
	for (size_t i = 0; i < length; ++i) {
		(*pResult)[i] = (*pResult)[i] ^ (*pInput)[i];
	}
}