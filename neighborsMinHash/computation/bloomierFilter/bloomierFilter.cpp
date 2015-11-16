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

vint* BloomierFilter::lookup(size_t pKey) {

}

void BloomierFilter::setValue(size_t pKey, size_t pValue) {

}
vint* BloomierFilter::findMatch(size_t pHashSeed, vint* pSubset) {

}

void BloomierFilter::create(std::map<size_t, size_t> pAssignment) {
	
}