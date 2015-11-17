#include "bloomierFilter.h"

BloomierFilter::BloomierFilter(size_t pHashSeed, std::map<size_t, size_t> pKeyDict, size_t pM, size_t pK, size_t pQ){
	mHashSeed = pHashSeed;
	mM = pM;;
	mK = pK;
	mQ = pQ;
	mKeyDict = pKeyDict;
	mBloomierHash = new BloomierHash(pHashSeed, pM, pK, pQ);
	mOrderAndMatchFinder = new OrderAndMatchFinder(pHashSeed, pKeyDict, pM, pK, pQ);
	vsize_t orderAndMatch = mOrderAndMatchFinder.find(); // datatype?
	mByteSize = this->getByteSize(pQ);
	mTable = new vvsize_t(pM, vsize_t(mByteSize, 0));
    mValueTable = new vsize_t(pM, 0);
    this->create(pKeyDict, orderAndMatch);
}

BloomierFilter::~BloomierFilter(){

}

vvsize_t* BloomierFilter::getTable() {
	return mTable;
}

void BloomierFilter::setTable(vvsize_t* pTable) {
	mTable = pTable;
}

vsize_t* BloomierFilter::getValueTable() {
	return mValueTablel
}
void BloomierFilter::setValueTable(vsize_t* pTable) {
	mValueTable = pTable;
} 

vsize_t* BloomierFilter::xorOperation(vsize_t* pValue, vsize_t* pM, vsize_t* pNeighbors) {
	this->byteArrayXor(pValue, pM);
	for (auto it = pNeighbors->begin(); it != pNeighbors->end(); ++it) {
		this->byteArrayXor(pValue, mTable[(*it)]);
	}
	return pValue;
}
vsize_t BloomierFilter::get(size_t pKey) {

}
void BloomierFilter::set(size_t pKey, size_t pValue) {

}

void BloomierFilter::create(std::map<size_t, size_t> pAssignment, OrderAndMatchFinder pPiTau) {

}

std::string BloomierFilter::tableToString() {

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