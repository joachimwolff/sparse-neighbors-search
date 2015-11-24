#include "bloomierFilter.h"

BloomierFilter::BloomierFilter(size_t pM, size_t pK, size_t pQ){
	// mHashSeed = pHashSeed;
	mM = pM;;
	mK = pK;
	mQ = pQ;
	// mKeyDict = pKeyDict;
	mBitVectorSize = ceil(pQ / static_cast<float>(CHAR_BIT));
	mBloomierHash = new BloomierHash(pM, pK, mBitVectorSize);
	// mOrderAndMatchFinder = new OrderAndMatchFinder(pHashSeed, pKeyDict, pM, pK, pQ);
	// vsize_t orderAndMatch = mOrderAndMatchFinder.find(); // datatype?
	// mByteSize = this->getByteSize(pQ);
	mTable = new bloomierTable(pM);
    mValueTable = new vvsize_t_p(pM);
    mEncoder = new Encoder(mBitVectorSize);
	mPiIndex = 0;
    // this->create(pKeyDict, orderAndMatch);
}

BloomierFilter::~BloomierFilter(){

}

bloomierTable* BloomierFilter::getTable() {
	return mTable;
}

void BloomierFilter::setTable(bloomierTable* pTable) {
	mTable = pTable;
}

vvsize_t_p* BloomierFilter::getValueTable() {
	return mValueTable;
}
void BloomierFilter::setValueTable(vvsize_t_p* pTable) {
	mValueTable = pTable;
} 

void BloomierFilter::xorOperation(bitVector* pValue, bitVector* pMask, vsize_t* pNeighbors) {
	this->xorBitVector(pValue, pMask);
	for (auto it = pNeighbors->begin(); it != pNeighbors->end(); ++it) {
		this->xorBitVector(pValue, (*mTable)[(*it)]);
	}
	// return pValue;
}
vsize_t* BloomierFilter::get(size_t pKey) {
	vsize_t* neighbors = mBloomierHash->getKNeighbors(pKey, mK, mM);
	bitVector* mask = mBloomierHash->getMask(pKey);

	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);

	size_t h = mEncoder->decode(valueToGet);
	if (h < neighbors->size()) {
		size_t L = (*neighbors)[h];
		if (L < mValueTable->size()) {
			return (*mValueTable)[L];
		}
	}
	return new vsize_t();
}
bool BloomierFilter::set(size_t pKey, size_t pValue) {
	vsize_t* neighbors = mBloomierHash->getKNeighbors(pKey, mK, mM);
	bitVector* mask = mBloomierHash->getMask(pKey);

	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);

	size_t h = mEncoder->decode(valueToGet);
	if (h < neighbors->size()) {
		size_t L = (*neighbors)[h];
		if (L < mValueTable->size()) {
			(*mValueTable)[L]->push_back(pValue);
			return true;
		}
	} else {
		vsize_t keys(1, pKey);
		vvsize_t values(1, vsize_t(1, pValue));
		this->create(&keys, &values, mPiIndex);
		return true;
	}
	return false;
}

void BloomierFilter::create(vsize_t* pKeys, vvsize_t* pValues, size_t piIndex) {
	
	mOrderAndMatchFinder->find(pKeys);
    vsize_t* piVector = mOrderAndMatchFinder->getPiVector();
	vsize_t* tauVector = mOrderAndMatchFinder->getTauVector();

	for (size_t i = piIndex; i < piVector->size(); ++i) {
		// size_t key = (*piList)[i];
		// size_t value = pAssignment[key];
		vsize_t* neighbors = mBloomierHash->getKNeighbors((*pKeys)[i], mK, mM);
		bitVector* mask = mBloomierHash->getMask((*pKeys)[i]);
		size_t l = (*tauVector)[i];
		size_t L = (*neighbors)[l];

		bitVector* encodeValue = mEncoder->encode(l);
		bitVector* valueToStore = new bitVector(mBitVectorSize, 0);
		this->xorBitVector(valueToStore, encodeValue);
		this->xorBitVector(valueToStore, mask);
		for (size_t j = 0; j < neighbors->size(); ++j) {
			if (j != l) {
				this->xorBitVector(valueToStore, (*mTable)[(*neighbors)[i]]);
			}
		}
		(*mTable)[L] = valueToStore;
		(*mValueTable)[L] = &(*pValues)[i];
	}
	mPiIndex = mPiIndex + pKeys->size();
	
}

void BloomierFilter::xorBitVector(bitVector* pResult, bitVector* pInput) {
	size_t length = std::min(pResult->size(), pInput->size());
	for (size_t i = 0; i < length; ++i) {
		(*pResult)[i] = (*pResult)[i] ^ (*pInput)[i];
	}
}