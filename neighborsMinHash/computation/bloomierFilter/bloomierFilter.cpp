#include "bloomierFilter.h"

BloomierFilter::BloomierFilter(std::map<size_t, size_t> pKeyDict, size_t pM, size_t pK, size_t pQ){
	mHashSeed = pHashSeed;
	mM = pM;;
	mK = pK;
	mQ = pQ;
	mKeyDict = pKeyDict;
	mBitVectorSize = ceil(pQ / static_cast<float>(CHAR_BIT));
	mBloomierHash = new BloomierHash(pHashSeed, pM, pK, mBitVectorSize);
	mOrderAndMatchFinder = new OrderAndMatchFinder(pHashSeed, pKeyDict, pM, pK, pQ);
	// vsize_t orderAndMatch = mOrderAndMatchFinder.find(); // datatype?
	// mByteSize = this->getByteSize(pQ);
	mTable = new bloomierTable(pM, bitVector(mBitVectorSize, 0));
    mValueTable = new vvsize_t_p(pM);
    mEncoder = new Encoder(mBitVectorSize);
    this->create(pKeyDict, orderAndMatch);
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
	return mValueTablel
}
void BloomierFilter::setValueTable(vvsize_t_p* pTable) {
	mValueTable = pTable;
} 

void BloomierFilter::xorOperation(bitVector* pValue, bitVector* pMask, vsize_t* pNeighbors) {
	this->xor(pValue, pMask);
	for (auto it = pNeighbors->begin(); it != pNeighbors->end(); ++it) {
		this->xor(pValue, (*mTable)[(*it)]);
	}
	return pValue;
}
vsize_t* BloomierFilter::get(size_t pKey) {
	vsize_t* neighbors = mBloomierHash.getNeighborhood(pKey);
	bitVector* mask = mBloomierHash.getM(pKey);

	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);

	size_t h = mEncoder.decode(valueToGet);
	if (h < neighbors->size()) {
		size_t L = (*neighbors)[h];
		if (L < mValueTable->size()) {
			return (*mValueTable)[L];
		}
	}
	return ;
}
bool BloomierFilter::set(size_t pKey, vsize_t* pValue) {
	vsize_t* neighbors = mBloomierHash.getNeighborhood(pKey);
	bitVector* mask = mBloomierHash.getM(pKey);

	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);

	size_t h = mEncoder.decode(valueToGet);
	if (h < neighbors->size()) {
		size_t L = (*neighbors)[h];
		if (L < mValueTable->size()) {
			(*mValueTable)[L] = pValue;
			return true;
		}
	}
	return false;
}

void BloomierFilter::create(vsize_t* pKeys, vvsize_t_p* pValues) {
	
	mOrderAndMatchFinder.find(pKeys);
    vsize_t* piVector = mOrderAndMatchFinder->getPiVector();
	vsize_t* tauVector = mOrderAndMatchFinder->getTauVector();

	for (size_t i = 0; i < piVector->size(); ++i) {
		// size_t key = (*piList)[i];
		// size_t value = pAssignment[key];
		vsize_t* neighbors = mBloomierHash.getNeighborhood((*pKeys)[i]);
		bitVector* mask = mBloomierHash.getMask((*pKeys)[i]);
		size_t l = (*tauVector)[i];
		size_t L = (*neighbors)[l];

		bitVector* encodeValue = mEncoder.encode(l);
		bitVector* valueToStore = new bitVector(mBitVectorSize);
		this->xor(valueToStore, encodeValue);
		this->xor(valueToStore, mask);
		for (size_t j = 0; j < neighbors->size(); ++j) {
			if (j != l) {
				this->xor(valueToStore, (*mTable)[(*neighbors)[i]]);
			}
		}
		(*mTable)[L] = valueToStore;
		(*mValueTable)[L] = (*pValues)[i];
	}
}

void BloomierFilter::xorBitVector(bitVector* pResult, bitVector* pInput) {
	size_t length = min(pResult->size(), pInput->size());
	for (size_t i = 0; i < length; ++i) {
		(*pResult)[i] = (*pResult)[i] ^ (*pInput)[i];
	}
}