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
		(*mTable)[i] = new bitVector(mBitVectorSize, 0);
	}
    mEncoder = new Encoder(mBitVectorSize);
	mPiIndex = 0; 
    mMaxBinSize = pMaxBinSize;
    mStoredNeighbors = new std::unordered_map<size_t, vsize_t* >();
}

BloomierFilter::~BloomierFilter() {
    delete mEncoder;
    for (size_t i = 0; i < mTable->size(); ++i) {
		delete (*mTable)[i];
	}
    delete mTable;
    delete mValueTable;
    delete mOrderAndMatchFinder;
    delete mBloomierHash;
    delete mStoredNeighbors; 
}
void BloomierFilter::xorOperation(bitVector* pValue, const bitVector* pMask, const vsize_t* pNeighbors) {
    this->xorBitVector(pValue, pMask);
	for (auto it = pNeighbors->begin(); it != pNeighbors->end(); ++it) {
		if (*it < mTable->size()) {
			this->xorBitVector(pValue, (*mTable)[(*it)]);
		}
	}
}
const vsize_t* BloomierFilter::get(const size_t pKey) {
    
    bool valueSeenBefor = mOrderAndMatchFinder->getValueSeenBefor(pKey);
    if (!valueSeenBefor) return NULL;
    vsize_t* neighbors = (*mStoredNeighbors)[pKey];
    
    if (neighbors == NULL) return NULL;
    const bitVector mask = mBloomierHash->getMask(pKey);
    
     
	bitVector valueToGet(mBitVectorSize, 0);
	this->xorOperation(&valueToGet, &mask, neighbors);
	const size_t h = mEncoder->decode(&valueToGet);
	if (h < neighbors->size()) {
		const size_t L = (*neighbors)[h];
		if (L < mValueTable->size()) {
            if ((*mValueTable)[L] != NULL) return (*mValueTable)[L];
            return NULL;
		}
	}
	return NULL;
}
bool BloomierFilter::set(const size_t pKey, const size_t pValue) {
    bool valueSeenBevor = mOrderAndMatchFinder->getValueSeenBefor(pKey);
    if (!valueSeenBevor) {
        this->create(pKey, pValue);
		return true;
    }
    vsize_t* neighbors = (*mStoredNeighbors)[pKey];
    if (neighbors == NULL) return false;
    
    const bitVector mask = mBloomierHash->getMask(pKey);
    
	bitVector valueToGet(mBitVectorSize, 0);
	this->xorOperation(&valueToGet, &mask, neighbors);
	const size_t h = mEncoder->decode(&valueToGet);

	if (h < neighbors->size()) {
		const size_t L = (*neighbors)[h];
		if (L < mValueTable->size()) {
            vsize_t* v = (*mValueTable)[L];
            if (v != NULL) {
                if (v->size() < mMaxBinSize) {
                    if (v->size() > 0) {
                        v->push_back(pValue);
                    }
                } else {
                    mOrderAndMatchFinder->deleteValueInBloomFilterInstance(pKey);
                    v->clear();
                }
            }
            return true;
		}
	}
	return false;
}

void BloomierFilter::create(const size_t pKey, const size_t pValue) {
	
    vsize_t* neighbors = mOrderAndMatchFinder->findIndexAndReturnNeighborhood(pKey);
    const vsize_t* piVector = mOrderAndMatchFinder->getPiVector();
	const vsize_t* tauVector = mOrderAndMatchFinder->getTauVector();
    (*mStoredNeighbors)[pKey] = neighbors;
    // const size_t key = (*piVector)[mPiIndex];
    const bitVector mask = mBloomierHash->getMask(pKey);
    const size_t l = (*tauVector)[mPiIndex];
    const size_t L = (*neighbors)[l];

    const bitVector* encodeValue = mEncoder->encode(l);
    this->xorBitVector((*mTable)[L], encodeValue);
    this->xorBitVector((*mTable)[L], &mask);
    
    for (size_t j = 0; j < neighbors->size(); ++j) {
        if (j != l) {
            this->xorBitVector((*mTable)[L], (*mTable)[(*neighbors)[j]]);
        }
    }
    if ((*mValueTable)[L] == NULL) {
            (*mValueTable)[L] = new vsize_t(1);
            (*mValueTable)[L]->operator[](0) = pValue;
    }
    delete encodeValue;
    ++mPiIndex;
}

void BloomierFilter::xorBitVector(bitVector* pResult, const bitVector* pInput) {
	const size_t length = std::min(pResult->size(), pInput->size());
	for (size_t i = 0; i < length; ++i) {
		(*pResult)[i] = (*pResult)[i] ^ (*pInput)[i];
	}
}