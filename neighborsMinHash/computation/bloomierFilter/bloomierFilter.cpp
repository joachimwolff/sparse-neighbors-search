#include "bloomierFilter.h"

BloomierFilter::BloomierFilter(std::map<size_t, size_t> pKeyDict, size_t pM, size_t pK, size_t pQ){
	mHashSeed = pHashSeed;
	mM = pM;;
	mK = pK;
	mQ = pQ;
	mKeyDict = pKeyDict;
	mBloomierHash = new BloomierHash(pHashSeed, pM, pK, pQ);
	mOrderAndMatchFinder = new OrderAndMatchFinder(pHashSeed, pKeyDict, pM, pK, pQ);
	// vsize_t orderAndMatch = mOrderAndMatchFinder.find(); // datatype?
	// mByteSize = this->getByteSize(pQ);
	mTable = new bloomierTable(pM, bitVector(mByteSize, 0));
    mValueTable = new vsize_t(pM, 0);
    mEncoder = new Encoder();
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

vsize_t* BloomierFilter::getValueTable() {
	return mValueTablel
}
void BloomierFilter::setValueTable(vsize_t* pTable) {
	mValueTable = pTable;
} 

vsize_t* BloomierFilter::xorOperation(vsize_t* pValue, bitVector* pMask, vsize_t* pNeighbors) {
	this->byteArrayXor(pValue, pMask);
	for (auto it = pNeighbors->begin(); it != pNeighbors->end(); ++it) {
		this->byteArrayXor(pValue, (*mTable)[(*it)]);
	}
	return pValue;
}
size_t BloomierFilter::get(size_t pKey) {
	vsize_t* neighbors = mBloomierHash.getNeighborhood(pKey);
	vsize_t* mask = mBloomierHash.getM(pKey);

	vsize_t* valueToGet = new vsize_t(mByteSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);

	size_t h = mEncoder.decode(valueToGet);
	if (h < neighbors->size()) {
		size_t L = (*neighbors)[h];
		if (L < mValueTable->size()) {
			return (*mValueTable)[L];
		}
	}
	return MAX_VALUE;
}
bool BloomierFilter::set(size_t pKey, size_t pValue) {
	vsize_t* neighbors = mBloomierHash.getNeighborhood(pKey);
	vsize_t* mask = mBloomierHash.getM(pKey);

	vsize_t* valueToGet = new vsize_t(mByteSize, 0);
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

void BloomierFilter::create(std::map<size_t, size_t> pAssignment) {
	mOrderAndMatchFinder.find();
    vsize_t* piVector = mOrderAndMatchFinder->getPiVector();
	vsize_t* tauVector = mOrderAndMatchFinder->getTauVector();

	for (size_t i = 0; i < piVector->size(); ++i) {
		size_t key = (*piList)[i];
		size_t value = pAssignment[key];
		vsize_t* neighbors = mBloomierHash.getNeighborhood(key);
		vsize_t* mask = mBloomierHash.getMask(key);
		size_t l = (*tauVector)[i];
		size_t L = (*neighbors)[l];

		vsize_t* encodeValue = mEncoder.encode(l, mByteSize);
		vsize_t* valueToStore = new vsize_t(mByteSize, 0);
		this->byteArrayXor(valueToStore, encodeValue);
		this->byteArrayXor(valueToStore, mask);
		for (size_t j = 0; j < neighbors->size(); ++j) {
			if (j != l) {
				this->byteArrayXor(valueToStore, (*mTable)[(*neighbors)[i]]);
			}
		}
		(*mTable)[L] = valueToStore;
		(*mValueTable)[L] = value;
	}
}

std::string BloomierFilter::tableToString() {
	// char* result = new char();
	// std::string result = 
}
std::pair<vsize_t, vsize_t > stringToTable(std::string pString) {

} 

size_t BloomierFilter::getByteSize(size_t pQ) {
	
}

void BloomierFilter::byteArrayXor(vsize_t* pResult, vsize_t* pInput) {
	size_t length = min(pResult->size(), pInput->size());
	for (size_t i = 0; i < length; ++i) {
		(*pResult)[i] = (*pResult)[i] ^ (*pInput)[i];
	}
}