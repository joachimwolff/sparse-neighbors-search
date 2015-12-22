#include "bloomierFilter.h"

BloomierFilter::BloomierFilter(const size_t pModulo, const size_t pNumberOfElements, const size_t pBitWidth, const size_t pHashSeed, const size_t pMaxBinSize){
	mModulo = pModulo;
	mNumberOfElements = pNumberOfElements;
    mHashSeed = pHashSeed;
	mBitVectorSize = ceil(pBitWidth / static_cast<float>(CHAR_BIT));
	mBloomierHash = new BloomierHash(pModulo, pNumberOfElements, mBitVectorSize, pHashSeed);
	mOrderAndMatchFinder = new OrderAndMatchFinder(pModulo, pNumberOfElements, mBloomierHash);
	mTable = new bloomierTable(pModulo);
	mValueTable = new vvsize_t_p(pModulo);
	for (size_t i = 0; i < pModulo; ++i) {
		(*mTable)[i] = new bitVector[mBitVectorSize];
	}
    mEncoder = new Encoder(mBitVectorSize);
	mPiIndex = 0; 
    mMaxBinSize = pMaxBinSize;
}

BloomierFilter::~BloomierFilter() {
    delete mEncoder;
    for (size_t i = 0; i < mTable->size(); ++i) {
		delete [] (*mTable)[i];
	}
    delete mTable;
    delete mValueTable;
    delete mOrderAndMatchFinder;
    delete mBloomierHash;
}
void BloomierFilter::xorOperation(bitVector* pValue, const bitVector* pMask, const vsize_t* pNeighbors) {
    for (size_t i = 0; i < mBitVectorSize; ++i) {
		pValue[i] = pMask[i];
	}
	for (auto it = pNeighbors->begin(); it != pNeighbors->end(); ++it) {
		if (*it < mTable->size()) {
			this->xorBitVector(pValue, (*mTable)[(*it)]);
		}
	}
}

void BloomierFilter::xorBitVector(bitVector* pResult, const bitVector* pInput) {
	for (size_t i = 0; i < mBitVectorSize; ++i) {
		pResult[i] = pResult[i] ^ pInput[i];
	}
}

const vsize_t* BloomierFilter::get(const size_t pKey) {
    
    unsigned char tries = 5;
    unsigned char i = 0;
    vsize_t* neighbors = new vsize_t(mNumberOfElements);
    const bitVector* mask = mBloomierHash->getMask(pKey);
    bitVector* valueToGet = new bitVector[mBitVectorSize];
    mBloomierHash->getKNeighbors(pKey, mOrderAndMatchFinder->getSeed(pKey), neighbors);
    this->xorOperation(valueToGet, mask, neighbors);
    const size_t h = mEncoder->decode(valueToGet, mBitVectorSize);
    if (h < neighbors->size()) {
        const size_t L = (*neighbors)[h];
        
        if (L < mValueTable->size()) {
            if ((*mValueTable)[L] != NULL) {
                delete neighbors;
                delete [] mask;
                delete [] valueToGet;
                return (*mValueTable)[L];
            }
        }
    }
    delete neighbors;
    delete [] mask;
    delete [] valueToGet;
    return NULL;
}
void BloomierFilter::set(const size_t pKey, const size_t pValue) {
    
    vsize_t* neighbors = new vsize_t(mNumberOfElements);
    const bitVector* mask = mBloomierHash->getMask(pKey);
    bitVector* valueToGet = new bitVector[mBitVectorSize];
    mBloomierHash->getKNeighbors(pKey, mOrderAndMatchFinder->getSeed(pKey), neighbors);
    
    this->xorOperation(valueToGet, mask, neighbors);
    const size_t h = mEncoder->decode(valueToGet, mBitVectorSize);
    if (h < neighbors->size()) {
        const size_t L = (*neighbors)[h];
        if (L < mValueTable->size()) {
            vsize_t* v = (*mValueTable)[L];
            if (v == NULL) {
                delete neighbors;
                delete [] mask;
                delete [] valueToGet;
                this->create(pKey, pValue);
                return;
            }
            if (v->size() < mMaxBinSize) {
                if (v->size() > 0) {
                    std::cout << __LINE__ << std::endl;
                    v->push_back(pValue);
                }
            } else {
                std::cout << __LINE__ << std::endl;
                
                v->clear();
            }
            delete neighbors;
            delete [] mask;
            delete [] valueToGet;
            return;
        }
    }
    delete neighbors;
    delete [] mask;
    delete [] valueToGet;
    this->create(pKey, pValue);
    return;
}

void BloomierFilter::create(const size_t pKey, const size_t pValue) {
    const vsize_t* neighbors = mOrderAndMatchFinder->findIndexAndReturnNeighborhood(pKey, mValueTable);
	const vsize_t* tauVector = mOrderAndMatchFinder->getTauVector();
    const bitVector* mask = mBloomierHash->getMask(pKey);
    const size_t l = (*tauVector)[mPiIndex];
    const size_t L = (*neighbors)[l];

    const bitVector* encodeValue = mEncoder->encode(l);
    
	for (size_t i = 0; i < mBitVectorSize; ++i) {
		(*mTable)[L][i] = encodeValue[i] ^ mask[i];
	}

    for (size_t j = 0; j < neighbors->size(); ++j) {
        if (j != l) {
            this->xorBitVector((*mTable)[L], (*mTable)[(*neighbors)[j]]);
            if ((*mValueTable)[(*neighbors)[j]] == NULL) {
                (*mValueTable)[j] = new vsize_t();
            }
        }
        
    }
    if ((*mValueTable)[L] == NULL) {
            (*mValueTable)[L] = new vsize_t(1);
            (*mValueTable)[L]->operator[](0) = pValue;
    } else {
        (*mValueTable)[L]->push_back(pValue);
    }
    delete [] encodeValue;
    delete [] mask;
    delete neighbors;
    ++mPiIndex;
}