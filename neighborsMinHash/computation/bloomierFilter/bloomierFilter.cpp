#include "bloomierFilter.h"

BloomierFilter::BloomierFilter(size_t pM, size_t pK, size_t pQ, size_t pHashSeed){
	mM = pM;;
	mK = pK;
	mQ = pQ;
	mBitVectorSize = ceil(pQ / static_cast<float>(CHAR_BIT));
	mBloomierHash = new BloomierHash(pM, pK, mBitVectorSize, pHashSeed);
	mOrderAndMatchFinder = new OrderAndMatchFinder(pM, pK, mBloomierHash);
	mTable = new bloomierTable(pM);
	mValueTable = new vvsize_t_p(pM);
	for (size_t i = 0; i < pM; ++i) {
		(*mTable)[i] = new bitVector(mBitVectorSize, 0);
		(*mValueTable)[i] = new vsize_t();
	}
	
    mEncoder = new Encoder(mBitVectorSize);
	mPiIndex = 0;
	// mOrderAndMatchFinder->find(pKeys);
	
}

BloomierFilter::~BloomierFilter(){

}
void BloomierFilter::check() {
	std::cout << __LINE__ << std::endl;
	size_t sumTable = 0;
	for(size_t i = 0; i < mTable->size(); ++i) {
        try {
		  sumTable += (*mTable)[i]->size();
        } catch(int e) {
            std::cout << "i: " << i << std::endl;
        }
	}
	std::cout << __LINE__ << std::endl;
	size_t sumValueTable = 0;
	for(size_t i = 0; i < mValueTable->size(); ++i) {
		sumValueTable += (*mValueTable)[i]->size();
	}
	std::cout << "sumTable: "  << sumTable << std::endl;
	std::cout << "sumValueTable: "  << sumValueTable << std::endl;
	
	std::cout << __LINE__ << std::endl;
	
}
bloomierTable* BloomierFilter::getTable() {
	return mTable;
}

void BloomierFilter::setTable(bloomierTable* pTable) {
    if (mTable != NULL) {
        for (size_t i = 0; i < mTable->size(); ++i) {
            delete (*mTable)[i];
        }
        delete mTable;
    }
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
		if (*it < mTable->size()) {
			this->xorBitVector(pValue, (*mTable)[(*it)]);
		}
	}
}
vsize_t* BloomierFilter::get(size_t pKey) {
    // std::cout << __LINE__ << std::endl;
	vsize_t* neighbors = mBloomierHash->getKNeighbors(pKey);
	bitVector* mask = mBloomierHash->getMask(pKey);
	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);

	size_t h = mEncoder->decode(valueToGet);
	if (h < neighbors->size()) {
		size_t L = (*neighbors)[h];
		if (L < mValueTable->size()) {
            delete neighbors;
			delete mask;
			delete valueToGet;
			return (*mValueTable)[L];
		}
	}
    delete neighbors;
	delete mask;
	delete valueToGet;
	return new vsize_t();
}
bool BloomierFilter::set(size_t pKey, size_t pValue) {
	vsize_t* neighbors = mBloomierHash->getKNeighbors(pKey);
	bitVector* mask = mBloomierHash->getMask(pKey);
	bitVector* valueToGet = new bitVector(mBitVectorSize, 0);
	this->xorOperation(valueToGet, mask, neighbors);
	size_t h = mEncoder->decode(valueToGet);
	
	delete mask;
	delete valueToGet;
	if (h < neighbors->size()) {
		size_t L = (*neighbors)[h];
		delete neighbors;
		if (L < mValueTable->size()) {
			vsize_t* v = ((*mValueTable)[L]);
			v->push_back(pValue);
			return true;
		}
	} else {
		delete neighbors;
		vsize_t* keys = new vsize_t (1, pKey);
		vvsize_t_p* values = new vvsize_t_p (1);
		(*values)[0] = new vsize_t(1, pValue);
		this->create(keys, values, mPiIndex);
		return true;
	}
	return false;
}

void BloomierFilter::create(vsize_t* pKeys, vvsize_t_p* pValues, size_t piIndex) {
	mOrderAndMatchFinder->find(pKeys);
    vsize_t* piVector = mOrderAndMatchFinder->getPiVector();
	vsize_t* tauVector = mOrderAndMatchFinder->getTauVector();
	for (size_t i = piIndex; i < piVector->size(); ++i) {
		vsize_t* neighbors = mBloomierHash->getKNeighbors((*pKeys)[i]);
		bitVector* mask = mBloomierHash->getMask((*pKeys)[i]);
		size_t l = (*tauVector)[i];
		size_t L = (*neighbors)[l];

		bitVector* encodeValue = mEncoder->encode(l);
		this->xorBitVector((*mTable)[L], encodeValue);
		this->xorBitVector((*mTable)[L], mask);
		for (size_t j = 0; j < neighbors->size(); ++j) {
			if (j != l) {
				this->xorBitVector((*mTable)[L], (*mTable)[(*neighbors)[j]]);
			}
		}
        for (size_t j = 0; j < (*pValues)[i-piIndex]->size(); ++j) {
            (*mValueTable)[L]->push_back((*pValues)[i-piIndex]->operator[](j));
        }
		delete neighbors;
		delete mask;
		delete encodeValue;
	}
	// mPiIndex += pKeys->size();
	delete pKeys;
	for (size_t i = 0; i < pValues->size(); ++i) {
		delete (*pValues)[i];
	}
	delete pValues;
	// delete piVector;
	// delete tauVector;
	
	
}

void BloomierFilter::xorBitVector(bitVector* pResult, bitVector* pInput) {
	size_t length = std::min(pResult->size(), pInput->size());
	for (size_t i = 0; i < length; ++i) {
		(*pResult)[i] = (*pResult)[i] ^ (*pInput)[i];
	}
}