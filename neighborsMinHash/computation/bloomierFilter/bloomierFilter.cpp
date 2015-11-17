#include "bloomierFilter.h"

BloomierFilter::BloomierFilter(size_t pHashSeed, size_t pM, size_t pK, size_t pQ){
	mHashSeed = pHashSeed;
	mM = pM;;
	mK = pK;
	mQ = pQ;
	mBloomierHash = new BloomierHash(pHashSeed, pM, pK, pQ);
	m;
}

BloomierFilter::~BloomierFilter(){

};

vsize_t* BloomierFilter::getTable() {

}

void BloomierFilter::setTable(vsize_t pTable) {

}

vsize_t* BloomierFilter::getValueTable() {

}
void BloomierFilter::setValueTable(vsize_t* pTable) {

} 

size_t BloomierFilter::xorOperation(size_t pValue, size_t pM, vsize_t pNeighbors) {

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