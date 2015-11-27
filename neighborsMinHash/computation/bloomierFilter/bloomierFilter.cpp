#include "bloomierFilter.h"

BloomierFilter::BloomierFilter(size_t pM, size_t pK, size_t pQ){
	// mHashSeed = pHashSeed;
	mM = pM;;
	mK = pK;
	mQ = pQ;
	// mKeyDict = pKeyDict;
	mBitVectorSize = ceil(pQ / static_cast<float>(CHAR_BIT));
	mBloomierHash = new BloomierHash(pM, pK, mBitVectorSize);
	mOrderAndMatchFinder = new OrderAndMatchFinder(pM, pK, pQ, mBloomierHash);
	// vsize_t orderAndMatch = mOrderAndMatchFinder.find(); // datatype?
	// mByteSize = this->getByteSize(pQ);
	mTable = new bloomierTable(pM);
	mValueTable = new vvsize_t_p(pM);
	for (size_t i = 0; i < pM; ++i) {
		(*mTable)[i] = new bitVector(mBitVectorSize, 0);
		(*mValueTable)[i] = new vsize_t();
	}
    // mValueTable = new vvsize_t_p(pM);
	
    mEncoder = new Encoder(mBitVectorSize);
	mPiIndex = 0;
    // this->create(pKeyDict, orderAndMatch);
}

BloomierFilter::~BloomierFilter(){

}
void BloomierFilter::check() {
	std::cout << __LINE__ << std::endl;
	
	for(size_t i = 0; i < mTable->size(); ++i) {
		(*mTable)[i]->size();
	}
	std::cout << __LINE__ << std::endl;
	
	for(size_t i = 0; i < mValueTable->size(); ++i) {
		(*mValueTable)[i]->size();
	}
	std::cout << __LINE__ << std::endl;
	
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
	// std::cout << "41" << std::endl;
	
	this->xorBitVector(pValue, pMask);
	// std::cout << "44" << std::endl;

	for (auto it = pNeighbors->begin(); it != pNeighbors->end(); ++it) {
	// std::cout << "47" << std::endl;

		this->xorBitVector(pValue, (*mTable)[(*it)]);
	// std::cout << "50" << std::endl;

	}
	// return pValue;
}
vsize_t* BloomierFilter::get(size_t pKey) {
	std::cout << __LINE__ << std::endl;
	
	check();
	std::cout << __LINE__ << std::endl;

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
	std::cout << __LINE__ << std::endl;

	check();
	std::cout << __LINE__ << std::endl;

	return new vsize_t();
}
bool BloomierFilter::set(size_t pKey, size_t pValue) {
	std::cout << __LINE__ << std::endl;
	check();
	std::cout << __LINE__ << std::endl;
	
	vsize_t* neighbors = mBloomierHash->getKNeighbors(pKey, mK, mM);
	bitVector* mask = mBloomierHash->getMask(pKey);
	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);
	size_t h = mEncoder->decode(valueToGet);
	std::cout << __LINE__ << std::endl;
	if (h < neighbors->size()) {
		size_t L = (*neighbors)[h];
		if (L < mValueTable->size()) {
				vsize_t* v = ((*mValueTable)[L]);
			if (v == NULL) {
				v = new vsize_t();
			}
	std::cout << __LINE__ << std::endl;
			v->push_back(pValue);
	std::cout << __LINE__ << std::endl;
			// (*mValueTable)[L] = v;
	std::cout << __LINE__ << std::endl;
			return true;
		}
	std::cout << __LINE__ << std::endl;
	} else {
	std::cout << __LINE__ << std::endl;
		vsize_t* keys = new vsize_t (1, pKey);
		vvsize_t_p* values = new vvsize_t_p (1);
		(*values)[0] = new vsize_t(1, pValue);
		this->create(keys, values, mPiIndex);
	std::cout << __LINE__ << std::endl;

		check();
	std::cout << __LINE__ << std::endl;

		return true;
	}
	std::cout << __LINE__ << std::endl;


	check();
	std::cout << __LINE__ << std::endl;

	return false;
}

void BloomierFilter::create(vsize_t* pKeys, vvsize_t_p* pValues, size_t piIndex) {
	std::cout << __LINE__ << std::endl;

	check();
	std::cout << __LINE__ << std::endl;

	mOrderAndMatchFinder->find(pKeys);
	// std::cout << "120" << std::endl;
	
    vsize_t* piVector = mOrderAndMatchFinder->getPiVector();
	// std::cout << "123" << std::endl;
	
	vsize_t* tauVector = mOrderAndMatchFinder->getTauVector();
	// std::cout << "126" << std::endl;

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
		(*mValueTable)[L] = (*pValues)[i];
	}
	mPiIndex = mPiIndex + pKeys->size();
	std::cout << __LINE__ << std::endl;

	check();
	std::cout << __LINE__ << std::endl;

}

void BloomierFilter::xorBitVector(bitVector* pResult, bitVector* pInput) {
	size_t length = std::min(pResult->size(), pInput->size());
	for (size_t i = 0; i < length; ++i) {
		(*pResult)[i] = (*pResult)[i] ^ (*pInput)[i];
	}
}